"""
test_amgs_stopping.jl

Tests for `AMGS_Stopping_Sampler`: the shared sampler contract, plus the
stopping rule itself (which is the only thing distinguishing it from base AMGS).
"""

@testset "AMGS_Stopping_Sampler" begin
    # `average_interval` is the window the acceptance rate is averaged over before
    # the stopping criterion is tested. Too short a window (e.g. 100) freezes the
    # kernel on a noisy estimate before the adaptation has converged, which is
    # valid but inefficient; 500 is a sensible working choice.
    make() = AMGS_Stopping_Sampler(0.234, 0.6, 20_000, 0.15, 0.35, 500)

    # The stopping rule freezes the kernel once the acceptance rate is inside
    # (0.15, 0.35), so the realised rate is only pinned to that window rather
    # than converging tightly on 0.234.
    test_sampler_contract("contract", make; target_accept=0.25, accept_atol=0.12)

    @testset "frozen kernel still samples the right target" begin
        # Freezing the proposal must not bias the chain: after adaptation stops
        # the kernel is a fixed Markov RWM kernel and must still be correct.
        ch = sample(
            StableRNG(SEED), corr_gaussian(), make(), 20_000;
            num_warmup=5_000, progress=false,
        )
        X = vecs(ch, @varname(x))
        @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=0.15)
        @test isapprox(var(X[1, :]), CORR_COV[1, 1]; atol=0.35)
        @test isapprox(var(X[2, :]), CORR_COV[2, 2]; atol=0.7)
    end

    @testset "adaptation actually stops" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = make()
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.stopped
    end

    @testset "frozen kernel stays frozen" begin
        # Once stopped, neither the covariance nor the global scale may move
        # again -- that is what makes the remaining chain a genuine Markov
        # random-walk Metropolis kernel.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = make()
        _, state = AbstractMCMC.step(rng, model, spl)
        while !state.stopped
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        Σ_frozen = copy(state.kernel)
        β_frozen = state.β
        step_at_stop = state.step

        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.kernel == Σ_frozen
        @test state.β == β_frozen
        @test state.stopped
        # ...but the chain must still be moving.
        @test state.step == step_at_stop + 2_000
    end

    @testset "stops at the interval boundary" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = make()
        _, state = AbstractMCMC.step(rng, model, spl)
        while !state.stopped
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.step % 500 == 0
    end

    @testset "max_adapt_period forces a stop" begin
        # An unreachable acceptance window means only the hard cap can stop it.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Stopping_Sampler(0.234, 0.6, 500, 0.999, 0.9999, 100)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test state.stopped
        @test state.step >= 500
    end

    @testset "max_adapt_period = -1 disables the cap" begin
        # With the cap disabled and an unreachable window, adaptation never stops.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = AMGS_Stopping_Sampler(0.234, 0.6, -1, 0.999, 0.9999, 100)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:2_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        @test !state.stopped
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError AMGS_Stopping_Sampler(0.234, 0.6, 100, 0.15, 0.35, 0)
        @test_throws ArgumentError AMGS_Stopping_Sampler(0.234, 0.6, 100, 0.4, 0.2, 100)
    end
end
