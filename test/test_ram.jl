"""
test_ram.jl

Tests for `RAM_Sampler` (Vihola's Robust Adaptive Metropolis).

RAM adapts the Cholesky factor directly, so the assertions target that factor
rather than a covariance/scale pair.
"""

@testset "RAM_Sampler" begin
    test_sampler_contract("contract", () -> RAM_Sampler(0.234, 0.6))

    @testset "drives the acceptance rate to its target" begin
        # RAM's defining property: the rank-one update provably steers the
        # acceptance rate to α* without estimating the target's mean.
        for target in (0.15, 0.234, 0.4)
            ch = sample(
                StableRNG(SEED), corr_gaussian(), RAM_Sampler(target, 0.6), 20_000;
                num_warmup=5_000, progress=false,
            )
            @test isapprox(accrate(ch), target; atol=0.05)
        end
    end

    @testset "Cholesky factor learns the target shape" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = RAM_Sampler(0.234, 0.9)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:20_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end

        @test state.kernel isa Cholesky
        Σ = Matrix(state.kernel)          # reconstruct S*S'
        @test issymmetric(Symmetric(Σ))
        @test isposdef(Symmetric(Σ))
        @test all(isfinite, Σ)

        # Shape, not scale: RAM folds the scale into the factor.
        ρ_adapted = Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2])
        ρ_target = CORR_COV[1, 2] / sqrt(CORR_COV[1, 1] * CORR_COV[2, 2])
        @test isapprox(ρ_adapted, ρ_target; atol=0.15)
    end

    @testset "kernel stays positive definite throughout" begin
        # Downdates are the risky direction: check every iteration survives.
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = RAM_Sampler(0.234, 0.6)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:3_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
            @test isposdef(Symmetric(Matrix(state.kernel)))
        end
    end

    @testset "eigenvalue bounds are respected when enabled" begin
        rng = StableRNG(SEED)
        model = corr_gaussian()
        spl = RAM_Sampler(0.234, 0.6; lb_eig=1e-6, ub_eig=5.0, bound_eigs=true)
        _, state = AbstractMCMC.step(rng, model, spl)
        for _ in 1:3_000
            _, state = AbstractMCMC.step(rng, model, spl, state)
        end
        λ = eigvals(Symmetric(Matrix(state.kernel)))
        @test all(1e-6 .<= λ .<= 5.0)
    end

    @testset "bounded and unbounded agree when bounds are wide" begin
        # Wide bounds must never reject an update, so the chains coincide.
        kw = (; num_warmup=200, progress=false)
        a = sample(StableRNG(7), corr_gaussian(), RAM_Sampler(0.234, 0.6), 500; kw...)
        b = sample(
            StableRNG(7), corr_gaussian(),
            RAM_Sampler(0.234, 0.6; lb_eig=-Inf, ub_eig=Inf, bound_eigs=true), 500; kw...,
        )
        @test vecs(a, @varname(x)) == vecs(b, @varname(x))
    end

    @testset "constructors" begin
        s = RAM_Sampler(0.234, 0.6)
        @test s.lb_eig == -Inf && s.ub_eig == Inf && s.bound_eigs == false
        s2 = RAM_Sampler(0.234, 0.6; lb_eig=1e-8, ub_eig=1e8, bound_eigs=true)
        @test s2.bound_eigs
        @test_throws ArgumentError RAM_Sampler(0.234, 0.6; lb_eig=5.0, ub_eig=1.0)
    end
end
