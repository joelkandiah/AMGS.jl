"""
test_amgs_sgs.jl

Tests for `AMGS_SGS_Sampler`, which switches between global-scaling adaptation
rules. Every rule is exercised against the same target so they can be compared
on equal terms.
"""

@testset "AMGS_SGS_Sampler" begin
    # :DA is the default; run the full contract on it.
    test_sampler_contract("contract (:DA)", () -> AMGS_SGS_Sampler(0.234, 0.6))

    # :NR uses a different scaling rule entirely, so it gets the contract too.
    test_sampler_contract(
        "contract (:NR)", () -> AMGS_SGS_Sampler(0.234, 0.6; method=:NR)
    )

    @testset "every method samples the target correctly" begin
        # :LOG is excluded from the acceptance-rate assertion because it can
        # only shrink the proposal (β = log(sigmoid(.)) < 0), so it cannot reach
        # an arbitrary target rate -- it is checked for correctness only.
        for method in (:RM, :DA, :EMA, :NR, :SINR, :CSINR, :LOG)
            ch = sample(
                StableRNG(SEED), corr_gaussian(),
                AMGS_SGS_Sampler(0.234, 0.6; method=method), 20_000;
                num_warmup=5_000, progress=false,
            )
            X = vecs(ch, @varname(x))
            @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=0.2)
            @test isapprox(var(X[1, :]), CORR_COV[1, 1]; atol=0.4)
            @test isapprox(var(X[2, :]), CORR_COV[2, 2]; atol=0.7)

            if method !== :LOG
                @test isapprox(accrate(ch), 0.234; atol=0.08)
            else
                # :LOG caps exp(β) at 1, so it cannot shrink the proposal enough
                # to reach an arbitrary target; it just must not diverge.
                @test 0.2 < accrate(ch) < 0.95
            end
        end
    end

    @testset ":EMA reproduces :RM exactly" begin
        # The EMA update expands to the RM update, so with the same seed the two
        # chains must be identical draw for draw.
        kw = (; num_warmup=500, progress=false)
        rm = sample(
            StableRNG(SEED), corr_gaussian(),
            AMGS_SGS_Sampler(0.234, 0.6; method=:RM), 1_000; kw...,
        )
        ema = sample(
            StableRNG(SEED), corr_gaussian(),
            AMGS_SGS_Sampler(0.234, 0.6; method=:EMA), 1_000; kw...,
        )
        # Identical in exact arithmetic; the two expressions differ only in
        # floating-point rounding, so compare approximately.
        @test isapprox(vecs(rm, @varname(x)), vecs(ema, @varname(x)); rtol=1e-6)
    end

    @testset ":LOG keeps the scale multiplier below one" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_SGS_Sampler(0.234, 0.6; method=:LOG)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
            @test state.β <= 0       # log(sigmoid(.)) is never positive
        end
        @test exp(state.β) <= 1
        # The unconstrained accumulator must be kept separately from the
        # reported log-scale, otherwise the rule diverges to -Inf.
        @test isfinite(state.global_scale_adaptation_state.u)
        @test state.β ≈ -log1p(exp(-state.global_scale_adaptation_state.u))
    end

    @testset "scale-invariant variance estimate tracks the acceptance error" begin
        # v̂ must accumulate (α - α*)^2. Once the sampler has converged to its
        # target the acceptance error is small, so v̂ must be small too -- this
        # is what fails if v̂ is built from the wrong reference quantity.
        for method in (:SINR, :CSINR)
            rng = StableRNG(SEED)
            model = corr_gaussian()
            spl = AMGS_SGS_Sampler(0.234, 0.6; method=method)
            _, state = AbstractMCMC.step(rng, model, spl)
            for _ in 1:20_000
                _, state = AbstractMCMC.step(rng, model, spl, state)
            end
            v̂ = state.global_scale_adaptation_state.v̂
            @test isfinite(v̂)
            @test 0 < v̂ < 0.25       # bounded by the variance of a probability
            @test isposdef(state.kernel)
        end
    end

    @testset "method state types are as expected" begin
        types = Dict(
            :RM => AMGS.RobbinsMonroState,
            :DA => AMGS.DualAveragingState,
            :EMA => AMGS.ExponentialMovingAverageState,
            :LOG => AMGS.LogisticRobbinsMonroState,
            :NR => AMGS.NewtonRaphsonRobbinsMonroState,
            :SINR => AMGS.ScaleInvariantNewtonRaphsonRobbinsMonroState,
            :CSINR => AMGS.ScaleInvariantNewtonRaphsonRobbinsMonroState,
        )
        for (method, T) in types
            rng = StableRNG(1)
            spl = AMGS_SGS_Sampler(0.234, 0.6; method=method)
            _, state = AbstractMCMC.step(rng, corr_gaussian(), spl)
            @test state.global_scale_adaptation_state isa T
            @test state.β == AMGS.get_beta(state.global_scale_adaptation_state)
        end
    end

    @testset "constructors" begin
        @test AMGS_SGS_Sampler(0.234, 0.6).method === Val(:DA)
        @test AMGS_SGS_Sampler(0.234, 0.6; method=:RM).method === Val(:RM)
        @test AMGS_SGS_Sampler(0.234, 0.6, true, 1e-4; method=:NR).use_noise_stabiliser ===
            Val(true)
        @test_throws ArgumentError AMGS_SGS_Sampler(0.234, 0.6; method=:nonsense)
        @test_throws ArgumentError AMGS_SGS_Sampler(0.234, 0.6; γ=0.0)
        @test_throws ArgumentError AMGS_SGS_Sampler(0.234, 0.6; κ=0.3)
    end
end
