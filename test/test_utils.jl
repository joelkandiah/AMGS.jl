"""
test_utils.jl

Shared models, chain accessors and reusable "sampler contract" test sets.

Every sampler in AMGS is a form of adaptive random-walk Metropolis operating on
the linked (real-valued) parameter vector, so they all satisfy the same
behavioural contract. `test_sampler_contract` encodes that contract once so each
sampler can be checked against it, rather than duplicating assertions per file.

MCMC tests are inherently stochastic. All chains are driven by `StableRNG` with
fixed seeds so results are reproducible across Julia versions, and tolerances
are set loose enough to be robust but tight enough to catch real regressions
(e.g. a missing Jacobian term, or adaptation that fails to reach its target).
"""

using AMGS
using Turing
using DynamicPPL
using FlexiChains
using Distributions
using LinearAlgebra
using MCMCChains
using Random
using StableRNGs
using Statistics
using Test
using AbstractMCMC

const SEED = 468

# ---------------------------------------------------------------- accessors --

"Stack a vector-valued parameter into a `d × N` matrix of draws."
vecs(ch, vn) = reduce(hcat, vec(ch[vn]))

"Extract a scalar parameter as a length-`N` vector of draws."
scal(ch, vn) = vec(ch[vn])

"Realised acceptance rate of the chain."
accrate(ch) = mean(vec(ch[:accepted]))

# ------------------------------------------------------------------- models --

"Correlated 2-D Gaussian with an analytically known mean and covariance."
@model function corr_gaussian()
    x ~ MvNormal([1.0, -2.0], [1.0 0.5; 0.5 2.0])
end

const CORR_MEAN = [1.0, -2.0]
const CORR_COV = [1.0 0.5; 0.5 2.0]

"""
Normal model with an `InverseGamma` variance.

The variance `s` has support `(0, ∞)`, so sampling it requires the link
transform *and* the corresponding log-Jacobian in the acceptance ratio. A
sampler that drops the Jacobian term still produces positive `s` but converges
to a visibly wrong posterior, so comparing against NUTS on this model is the
key regression test for the transform handling.
"""
@model function normal_invgamma(y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    return y .~ Normal(m, sqrt(s))
end

"Hierarchical model containing a vector-valued block, for Gibbs tests."
@model function vector_block(y)
    s ~ InverseGamma(2, 3)
    b ~ MvNormal(zeros(3), I)
    return y .~ Normal(sum(b), sqrt(s))
end

"Deterministic dataset shared across tests."
const YDATA = randn(StableRNG(7), 60) .* 1.5 .+ 0.8

# ------------------------------------------------------- the shared contract --

"""
    test_sampler_contract(name, make_sampler; kwargs...)

Run the behavioural contract shared by every AMGS sampler.

`make_sampler` is a zero-argument callable returning a fresh sampler instance.

# Keyword arguments
- `target_accept`: the acceptance rate the sampler is expected to adapt towards,
  or `nothing` to skip that check (samplers that deliberately do not target a
  fixed rate, such as tempered variants during their schedule).
- `n_samples` / `n_warmup`: chain lengths. Defaults are sized to keep the suite
  quick while leaving Monte Carlo error well inside the tolerances.
- `gibbs`: whether to run the Gibbs-composition tests.
- `mean_atol` / `var_atol`: tolerances for the Gaussian recovery test.
"""
function test_sampler_contract(
    name::String,
    make_sampler;
    target_accept::Union{Nothing,Real}=0.234,
    accept_atol::Real=0.06,
    n_samples::Int=20_000,
    n_warmup::Int=5_000,
    gibbs::Bool=true,
    mean_atol::Real=0.15,
    var_atol::Real=0.35,
)
    @testset "$name" begin

        # -- 1. Recovers a known unconstrained target ----------------------
        @testset "recovers correlated Gaussian" begin
            ch = sample(
                StableRNG(SEED), corr_gaussian(), make_sampler(), n_samples;
                num_warmup=n_warmup, progress=false,
            )
            X = vecs(ch, @varname(x))
            @test size(X, 2) == n_samples
            @test isapprox(vec(mean(X; dims=2)), CORR_MEAN; atol=mean_atol)
            @test isapprox(var(X[1, :]), CORR_COV[1, 1]; atol=var_atol)
            @test isapprox(var(X[2, :]), CORR_COV[2, 2]; atol=2 * var_atol)
            @test isapprox(cov(X[1, :], X[2, :]), CORR_COV[1, 2]; atol=var_atol)

            # The chain must actually move; a stuck chain would pass a mean test
            # by luck but has no unique draws.
            @test length(unique(X[1, :])) > n_samples ÷ 20
        end

        # -- 2. Adaptation reaches the target acceptance rate --------------
        if target_accept !== nothing
            @testset "adapts to target acceptance rate" begin
                ch = sample(
                    StableRNG(SEED), corr_gaussian(), make_sampler(), n_samples;
                    num_warmup=n_warmup, progress=false,
                )
                @test isapprox(accrate(ch), target_accept; atol=accept_atol)
            end
        end

        # -- 3. Constrained support: link transform + Jacobian -------------
        @testset "constrained support matches NUTS" begin
            mdl = normal_invgamma(YDATA)
            ch = sample(
                StableRNG(SEED), mdl, make_sampler(), n_samples;
                num_warmup=n_warmup, progress=false,
            )
            s = scal(ch, @varname(s))
            m = scal(ch, @varname(m))

            # Constraint is respected for every single draw.
            @test all(s .> 0)
            @test all(isfinite, s) && all(isfinite, m)

            ref = sample(StableRNG(SEED), mdl, NUTS(), 4_000; progress=false)
            s_ref = scal(ref, @varname(s))
            m_ref = scal(ref, @varname(m))

            # Agreement with NUTS is what catches a dropped log-Jacobian.
            @test isapprox(mean(m), mean(m_ref); atol=0.12)
            @test isapprox(mean(s), mean(s_ref); atol=0.35)
            @test isapprox(std(m), std(m_ref); atol=0.12)
        end

        # -- 4. Gibbs composition, including vector-valued blocks ----------
        if gibbs
            @testset "Gibbs with vector-valued block" begin
                mdl = vector_block(YDATA)
                spl = Turing.Gibbs(
                    @varname(s) => make_sampler(), @varname(b) => make_sampler()
                )
                ch = sample(
                    StableRNG(SEED), mdl, spl, 6_000; num_warmup=2_000, progress=false
                )
                B = vecs(ch, @varname(b))
                @test size(B, 1) == 3          # vector block kept its shape
                @test size(B, 2) == 6_000
                @test all(scal(ch, @varname(s)) .> 0)

                ref = sample(StableRNG(SEED), mdl, NUTS(), 3_000; progress=false)
                B_ref = vecs(ref, @varname(b))
                @test isapprox(
                    mean(sum(B; dims=1)), mean(sum(B_ref; dims=1)); atol=0.2
                )
            end

            @testset "Gibbs composed with NUTS" begin
                mdl = vector_block(YDATA)
                spl = Turing.Gibbs(
                    @varname(s) => make_sampler(), @varname(b) => NUTS()
                )
                ch = sample(
                    StableRNG(SEED), mdl, spl, 2_000; num_warmup=1_000, progress=false
                )
                @test size(vecs(ch, @varname(b)), 2) == 2_000
                @test all(scal(ch, @varname(s)) .> 0)
            end
        end

        # -- 4b. Emitted transitions are self-consistent --------------------
        @testset "reported log-density matches reported parameters" begin
            # Catches stale cached transitions: every emitted draw's reported
            # log-density must be the log-density *of the parameters reported
            # alongside it*. This is checked analytically against the known
            # Gaussian target, and is the assertion that fails if a sampler
            # lets its transition cache drift out of step with its position
            # (e.g. by not maintaining it while warmup samples are discarded).
            ch = sample(
                StableRNG(SEED), corr_gaussian(), make_sampler(), 500;
                num_warmup=200, progress=false,
            )
            X = vecs(ch, @varname(x))
            lj = vec(ch[:logjoint])
            target = MvNormal(CORR_MEAN, CORR_COV)
            expected = [logpdf(target, X[:, i]) for i in axes(X, 2)]
            @test isapprox(lj, expected; rtol=1e-8)
        end

        # -- 5. Reproducibility --------------------------------------------
        @testset "reproducible under a fixed seed" begin
            kw = (; num_warmup=200, progress=false)
            a = sample(StableRNG(99), corr_gaussian(), make_sampler(), 500; kw...)
            b = sample(StableRNG(99), corr_gaussian(), make_sampler(), 500; kw...)
            @test vecs(a, @varname(x)) == vecs(b, @varname(x))

            c = sample(StableRNG(100), corr_gaussian(), make_sampler(), 500; kw...)
            @test vecs(a, @varname(x)) != vecs(c, @varname(x))
        end

        # -- 6. Chain backends ----------------------------------------------
        @testset "chain backends" begin
            kw = (; num_warmup=200, progress=false)
            fc = sample(StableRNG(1), corr_gaussian(), make_sampler(), 500; kw...)
            @test fc isa FlexiChains.VNChain           # FlexiChains is the default

            mc = sample(
                StableRNG(1), corr_gaussian(), make_sampler(), 500;
                chain_type=MCMCChains.Chains, kw...,
            )
            @test mc isa MCMCChains.Chains
            @test size(mc, 1) == 500
        end
    end
end
