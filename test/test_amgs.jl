"""
test_amgs.jl

Tests for the base `AMGS_Sampler`: the shared sampler contract plus checks of
the adaptation internals that are specific to this sampler.
"""

@testset "AMGS_Sampler" begin
    test_sampler_contract("contract", () -> AMGS_Sampler(0.234, 0.6))

    @testset "adapts towards different target rates" begin
        # The Robbins-Monro scale adaptation should track whatever target it is
        # given, not just the 0.234 default.
        for target in (0.15, 0.4)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), AMGS_Sampler(target, 0.6), 20_000;
                num_warmup=5_000, progress=false,
            )
            @test isapprox(accrate(ch), target; atol=0.06)
        end
    end

    @testset "covariance adaptation learns the target correlation" begin
        # After adaptation the proposal covariance should be proportional to the
        # target covariance -- the entire point of Haario-style adaptation.
        # Driven through `AbstractMCMC.step` directly so we can inspect the
        # sampler state, which the chain itself does not retain.
        # Uses γ = 0.9 rather than 0.6: the Robbins-Monro gain (t+1)^-γ decays
        # faster, so the adapted covariance is a much lower-variance estimator
        # (sd of the adapted correlation is ~0.02 across seeds, versus ~0.05 at
        # γ = 0.6). Both are unbiased -- this just makes the assertion stable
        # rather than seed-dependent.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Sampler(0.234, 0.9)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end

        Σ = state.kernel
        @test size(Σ) == (2, 2)
        @test issymmetric(Σ)
        @test isposdef(Σ)
        @test all(isfinite, Σ)

        # Compare correlation rather than scale, since `kernel` is defined up to
        # the separate `exp(β)` scale factor.
        ρ_adapted = Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2])
        ρ_target = CORR_COV[1, 2] / sqrt(CORR_COV[1, 1] * CORR_COV[2, 2])
        @test isapprox(ρ_adapted, ρ_target; atol=0.1)

        # The running mean should track the target mean.
        @test isapprox(state.μ, CORR_MEAN; atol=0.2)
    end

    @testset "noise stabiliser keeps the kernel positive definite" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Sampler(0.234, 0.6, true, 1e-6)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test isposdef(state.kernel)
        @test all(isfinite, state.kernel)
    end

    @testset "warmup steps advance the adaptation" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Sampler(0.234, 0.6)
        _, state = AbstractMCMC.step_warmup(rng, model, spl)
        step0 = state.step
        for _ in 1:500
            _, state = AbstractMCMC.step_warmup(rng, model, spl, state)
        end
        @test state.step == step0 + 500

        # `discard_sample=true` must emit no transition but still adapt.
        t, state2 = AbstractMCMC.step_warmup(
            rng, model, spl, state; discard_sample=true
        )
        @test t === nothing
        @test state2.step == state.step + 1
    end

    @testset "constructors" begin
        s1 = AMGS_Sampler(0.234, 0.6)
        @test s1.use_noise_stabiliser === Val(false)
        @test s1.acceptance_rate == 0.234
        @test s1.convergence_step == 0.6

        s2 = AMGS_Sampler(0.234, 0.6, true, 1e-4)
        @test s2.use_noise_stabiliser === Val(true)
        @test s2.noise_scaling == 1e-4
    end
end
