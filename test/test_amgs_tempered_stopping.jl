"""
test_amgs_tempered_stopping.jl

Tests for `AMGS_Stopping_Tempered_Sampler`, which combines tempering with the
adaptation stopping rule. Both features are checked, plus their interaction:
freezing the kernel must not disturb the tempered stationary distribution.
"""

@testset "AMGS_Stopping_Tempered_Sampler" begin
    # Default temperature is 1, and the default stopping window (1.01, 1.02) is
    # unreachable, so this configuration is plain AMGS and must satisfy the
    # full contract.
    test_sampler_contract(
        "contract (temperature = 1, no stopping)",
        () -> AMGS_Stopping_Tempered_Sampler(0.234, 0.6),
    )

    @testset "default stopping window is unreachable" begin
        # Acceptance rates cannot exceed 1, so the (1.01, 1.02) defaults mean
        # adaptation runs forever unless the user opts in.
        rng = StableRNG(SEED)
        spl = AMGS_Stopping_Tempered_Sampler(0.234, 0.6)
        _, state = AbstractMCMC.step(rng, corr_gaussian(), spl)
        for _ in 1:3_000
            _, state = AbstractMCMC.step(rng, corr_gaussian(), spl, state)
        end
        @test !state.stopped
    end

    @testset "opting in to acceptance-based stopping freezes the kernel" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6;
            lower_stopping_accep=0.15,
            upper_stopping_accep=0.35,
            adaptation_interval=500,
        )
        _, state = AbstractMCMC.step(rng, model, spl)
        while !state.stopped
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.step % 500 == 0
        Σ_frozen = copy(state.kernel)
        β_frozen = state.β
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.kernel == Σ_frozen
        @test state.β == β_frozen
    end

    @testset "max_adaptation forces a stop" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6; adaptation_interval=100, max_adaptation=500
        )
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.stopped
        @test state.step >= 500
    end

    @testset "tempering rescales a Gaussian target by 1/temp" begin
        for temp in (0.5, 2.0)
            spl = AMGS_Stopping_Tempered_Sampler(0.234, 0.6; temperature=temp)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), spl, 30_000;
                num_warmup=10_000, progress=false,
            )
            X = vecs(ch, @varname(x))
            expected = CORR_COV ./ temp
            @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=0.2 / sqrt(temp))
            @test isapprox(var(X[1, :]), expected[1, 1]; rtol=0.25)
            @test isapprox(var(X[2, :]), expected[2, 2]; rtol=0.25)
        end
    end

    @testset "tempering and stopping compose correctly" begin
        # With the kernel frozen, the tempered target must still be sampled
        # correctly -- freezing changes efficiency, not the stationary law.
        spl = AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6; temperature=0.5,
            lower_stopping_accep=0.15, upper_stopping_accep=0.35,
            adaptation_interval=500,
        )
        ch = sample(
            StableRNG(SEED), corr_gaussian(), spl, 30_000;
            num_warmup=10_000, progress=false,
        )
        X = vecs(ch, @varname(x))
        expected = CORR_COV ./ 0.5
        @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=0.3)
        @test isapprox(var(X[1, :]), expected[1, 1]; rtol=0.3)
        @test isapprox(var(X[2, :]), expected[2, 2]; rtol=0.3)
    end

    @testset "constructors" begin
        s = AMGS_Stopping_Tempered_Sampler(0.234, 0.6)
        @test s.temperature == 1.0
        @test s.adaptation_interval == 200
        @test s.max_adaptation == -1
        @test s.lower_stopping_accep == 1.01

        s2 = AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6, true, 1e-4; temperature=0.5, adaptation_interval=50
        )
        @test s2.use_noise_stabiliser === Val(true)
        @test s2.temperature == 0.5
        @test s2.adaptation_interval == 50

        @test_throws ArgumentError AMGS_Stopping_Tempered_Sampler(0.234, 0.6; temperature=0)
        @test_throws ArgumentError AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6; adaptation_interval=0
        )
        @test_throws ArgumentError AMGS_Stopping_Tempered_Sampler(
            0.234, 0.6; lower_stopping_accep=0.5, upper_stopping_accep=0.2
        )
    end
end
