"""
test_amgs_da.jl

Tests for `AMGS_DA_Sampler` (dual-averaging adaptation of the global scale).
"""

@testset "AMGS_DA_Sampler" begin
    test_sampler_contract("contract", () -> AMGS_DA_Sampler(0.234, 0.6))

    @testset "dual averaging drives the acceptance rate to its target" begin
        for target in (0.15, 0.234, 0.4)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), AMGS_DA_Sampler(target, 0.6), 20_000;
                num_warmup=5_000, progress=false,
            )
            @test isapprox(accrate(ch), target; atol=0.07)
        end
    end

    @testset "dual-averaging accumulators stay finite and converge" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_DA_Sampler(0.234, 0.6)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test isfinite(state.x̄)
        @test isfinite(state.ḡ)
        @test state.β == state.x̄          # the averaged iterate is the live scale
        # ḡ is the running average of (target - realised acceptance); once the
        # scale has converged this error should be small.
        @test abs(state.ḡ) < 0.05
        @test isposdef(state.kernel)
    end

    @testset "acceptance rate is stable across the chain" begin
        # What dual averaging guarantees is that the *time-averaged* acceptance
        # error vanishes -- not that β itself settles. β keeps moving because
        # the Robbins-Monro covariance is a deliberately live, noisy estimate at
        # γ = 0.6, and the scale tracks it; only the product exp(β)Σ is pinned
        # down. So the meaningful check is that the sampler's realised behaviour
        # is the same in the first and second halves of a long chain.
        ch = sample(
            StableRNG(SEED), corr_gaussian(), AMGS_DA_Sampler(0.234, 0.6), 40_000;
            num_warmup=5_000, progress=false,
        )
        acc = vec(ch[:accepted])
        h1 = mean(acc[1:20_000])
        h2 = mean(acc[20_001:end])
        @test isapprox(h1, 0.234; atol=0.05)
        @test isapprox(h2, 0.234; atol=0.05)
        @test isapprox(h1, h2; atol=0.05)
    end

    @testset "constructors" begin
        s = AMGS_DA_Sampler(0.234, 0.6)
        @test s.μ == 0.0 && s.γ == 0.05 && s.t0 == 10.0 && s.κ == 0.75
        @test s.use_noise_stabiliser === Val(false)

        s2 = AMGS_DA_Sampler(0.234, 0.6, true, 1e-4; γ=0.1, κ=0.9)
        @test s2.use_noise_stabiliser === Val(true)
        @test s2.γ == 0.1 && s2.κ == 0.9

        @test_throws ArgumentError AMGS_DA_Sampler(0.234, 0.6; γ=0.0)
        @test_throws ArgumentError AMGS_DA_Sampler(0.234, 0.6; t0=0.0)
        @test_throws ArgumentError AMGS_DA_Sampler(0.234, 0.6; κ=0.4)
        @test_throws ArgumentError AMGS_DA_Sampler(0.234, 0.6; κ=1.5)
    end
end
