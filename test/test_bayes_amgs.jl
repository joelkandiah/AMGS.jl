"""
test_bayes_amgs.jl

Tests for `Bayes_AMGS_Sampler`: the Bayesian (inverse-Wishart posterior mean)
covariance estimate over a sliding history window, plus the multiplicative
scale adaptation with its restart rule.
"""

@testset "Bayes_AMGS_Sampler" begin
    # Window function: drop the oldest half of the history. Non-decreasing,
    # steps by at most 1, and n - f(n) -> infinity as required.
    halfwindow(n) = n ÷ 2
    make() = Bayes_AMGS_Sampler(0.234, 5.0, 1e-6, halfwindow)

    test_sampler_contract("contract", make; accept_atol=0.08)

    @testset "scale adaptation reaches the target acceptance rate" begin
        for target in (0.15, 0.234, 0.4)
            spl = Bayes_AMGS_Sampler(target, 5.0, 1e-6, halfwindow)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), spl, 20_000;
                num_warmup=5_000, progress=false,
            )
            @test isapprox(accrate(ch), target; atol=0.08)
        end
    end

    @testset "windowed covariance learns the target shape" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = make()
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        S = state.kernel
        @test isposdef(Symmetric(S))
        @test all(isfinite, S)

        ρ_adapted = S[1, 2] / sqrt(S[1, 1] * S[2, 2])
        ρ_target = CORR_COV[1, 2] / sqrt(CORR_COV[1, 1] * CORR_COV[2, 2])
        @test isapprox(ρ_adapted, ρ_target; atol=0.15)

        # The window mean should track the target mean.
        @test isapprox(state.μ, CORR_MEAN; atol=0.3)
    end

    @testset "scale stays above its floor and finite" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = Bayes_AMGS_Sampler(0.234, 5.0, 0.05, halfwindow)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:5_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
            @test state.λ >= 0.05
            @test isfinite(state.λ)
        end
    end

    @testset "steplength constants match the paper" begin
        # c = 2.38^2/d, n_start = 5/(a(1-a)), A = -Φ⁻¹(a/2).
        for (a, d) in ((0.234, 2), (0.4, 5))
            n_start, A, δ, c = AMGS.bayes_constants(a, d)
            @test c ≈ 2.38^2 / d
            @test n_start ≈ 5 / (a * (1 - a))
            @test A ≈ -quantile(Normal(0, 1), a / 2)
            @test δ > 0
        end
    end

    @testset "window function is validated" begin
        # `f` must not jump by more than 1 per step; a jump leaves the sliding
        # window update undefined, so it must be reported rather than silently
        # producing a wrong covariance.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        bad = Bayes_AMGS_Sampler(0.234, 5.0, 1e-6, n -> 3 * n)
        _, state = AbstractMCMC.step(rng, model, bad)
        @test_throws ArgumentError AbstractMCMC.step(rng, model, bad, state)
    end

    @testset "a constant window keeps the whole history" begin
        # f(n) = 0 never drops anything, so only the rank-one add branch runs.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = Bayes_AMGS_Sampler(0.234, 5.0, 1e-6, _ -> 0)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test isposdef(Symmetric(state.kernel))
        @test isapprox(state.μ, CORR_MEAN; atol=0.4)
    end

    @testset "get_index_ll indexes the history" begin
        ll = AMGS.cons([3.0], AMGS.cons([2.0], AMGS.cons([1.0], AMGS.nil(Vector{Float64}))))
        @test AMGS.get_index_ll(1, ll) == [3.0]
        @test AMGS.get_index_ll(2, ll) == [2.0]
        @test AMGS.get_index_ll(3, ll) == [1.0]
        @test_throws ArgumentError AMGS.get_index_ll(0, ll)
    end

    @testset "constructors" begin
        s = make()
        @test s.ν₀ == 5.0
        @test s.λ_min == 1e-6
        @test s.use_noise_stabiliser == false
        s2 = Bayes_AMGS_Sampler(0.234, 5.0, 1e-6, halfwindow; use_noise_stabiliser=true)
        @test s2.use_noise_stabiliser
        @test_throws ArgumentError Bayes_AMGS_Sampler(0.0, 5.0, 1e-6, halfwindow)
        @test_throws ArgumentError Bayes_AMGS_Sampler(1.0, 5.0, 1e-6, halfwindow)
        @test_throws ArgumentError Bayes_AMGS_Sampler(0.234, 5.0, 0.0, halfwindow)
    end
end
