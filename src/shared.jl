"""
shared.jl

Machinery common to every sampler in this package.

All AMGS samplers are adaptive random-walk Metropolis kernels. They differ only
in *how* they adapt (Robbins–Monro, dual averaging, RAM rank-one updates, …) and
in *what* they target (the posterior, or a tempered version of it). Everything
else — building the linked log-density, drawing the Gaussian proposal, forming
transitions, and plugging into Turing's Gibbs machinery — is identical, and
lives here so it is implemented and tested once.

## The unconstrained-space contract

Every sampler here is a Gaussian random walk, which is only valid on a
real-valued space. That is enforced structurally: samplers never touch a
`VarInfo` directly, they hold a `DynamicPPL.LogDensityFunction` built with
`DynamicPPL.LinkAll()`, so the parameter vector `θ` is always the *linked*
(unconstrained) representation. Constrained variables — variances, simplices,
bounded parameters — are transformed to ℝⁿ automatically, and the log-Jacobian
of that transform is carried inside the target density. Dropping the Jacobian
would leave the chain converging to the wrong distribution, so the target is
always one of:

- `DynamicPPL.getlogjoint_internal` = log prior + log Jacobian + log likelihood,
- [`TemperedLogJoint`](@ref) = log prior + log Jacobian + `temp` × log likelihood.

!!! warning
    Use `getlogprior_internal` (which already includes the Jacobian with the
    correct sign) when composing a custom target. `getlogjac` follows the
    opposite sign convention, so `getlogprior + getlogjac` is *not* the same
    thing and silently biases the chain on constrained models.
"""

"""
    AbstractAMGSState

Supertype of every sampler state in this package.

Subtypes are expected to provide at least the fields `ldf`, `transition`, `θ`,
`logp`, `μ`, `kernel`, `step`, `β`, `Δ_cache`, `θ_prop` and `z_cache`; the
generic helpers below are written against that interface so each sampler only
has to implement its own adaptation rule.
"""
abstract type AbstractAMGSState end

# ------------------------------------------------------------------ targets --

"""
    TemperedLogJoint(temp)

Callable target density for the tempered samplers.

Tempering is applied to the posterior **on the constrained space**, and the
result is then transformed to the unconstrained space the sampler works in:

    log π_temp(θ) = temp · (log p(θ) + log p(y | θ)) + log|J(θ)|

so both the prior and the likelihood are flattened together by `temp`, while the
log-Jacobian of the link transform is *not* tempered — it belongs to the change
of variables, not to the target. `temp < 1` flattens the posterior and improves
mixing between modes; `temp > 1` sharpens it. At `temp = 1` this reduces exactly
to `DynamicPPL.getlogjoint_internal`.

!!! note "Sign of the Jacobian term"
    `DynamicPPL.getlogjac` returns the Jacobian with the *opposite* sign to the
    one the internal log-densities add: `getlogprior_internal == getlogprior -
    getlogjac`. Hence the `- getlogjac` below is what adds `+log|J|`. Writing
    `getlogprior + getlogjac` instead would flip the Jacobian and silently bias
    the chain on any constrained model.
"""
struct TemperedLogJoint{T<:Real}
    temp::T
end

function (f::TemperedLogJoint)(oavi)
    return f.temp *
           (DynamicPPL.getlogprior(oavi) + DynamicPPL.getloglikelihood(oavi)) -
           DynamicPPL.getlogjac(oavi)
end

"The accumulators required to evaluate a prior/Jacobian/likelihood decomposition."
const TEMPERED_ACCS = (
    DynamicPPL.LogPriorAccumulator(),
    DynamicPPL.LogJacobianAccumulator(),
    DynamicPPL.LogLikelihoodAccumulator(),
)

# ------------------------------------------------------- log-density set-up --

"""
    build_ldf(model, rng, init_strategy, [getlogdensity])

Build the `LogDensityFunction` a sampler runs against: the model's log-density
over the fully *linked* parameter vector.

Returns the `LogDensityFunction`; its initial parameter vector is available as
[`initial_theta`](@ref).
"""
function build_ldf(
    model::DynamicPPL.Model,
    rng::AbstractRNG,
    init_strategy,
    getlogdensity=DynamicPPL.getlogjoint_internal,
)
    oavi = DynamicPPL.OnlyAccsVarInfo(DynamicPPL.VectorValueAccumulator())
    _, oavi = DynamicPPL.init!!(rng, model, oavi, init_strategy, DynamicPPL.LinkAll())
    vecvals = DynamicPPL.get_vector_values(oavi)
    accs = if getlogdensity === DynamicPPL.getlogjoint_internal
        DynamicPPL.ldf_accs(getlogdensity)
    else
        TEMPERED_ACCS
    end
    return DynamicPPL.LogDensityFunction(model, getlogdensity, vecvals, accs)
end

"""
    initial_theta(ldf)

The linked parameter vector the `LogDensityFunction` was constructed at. Copied,
since samplers mutate `θ` in place.
"""
initial_theta(ldf::DynamicPPL.LogDensityFunction) = copy(ldf._x)

"""
    initial_kernel(θ)

Initial proposal covariance: `diag(0.01|θᵢ| + 1e-6)`, i.e. scaled to the
magnitude of the starting point, with a floor that keeps the matrix positive
definite when a component starts at exactly zero. Only a starting guess — the
adaptation takes over from the first step.
"""
initial_kernel(θ::AbstractVector) = diagm(0.01 .* abs.(θ) .+ 1e-6)

# ------------------------------------------------------------------ noise ----

"""
    add_noise!(rng, Σ, noise_scaling, n, use_noise_stabiliser)

Add small random positive noise to the diagonal of the adapted covariance when
the stabiliser is enabled.

In high dimensions the empirical covariance can become numerically singular,
which makes the Cholesky factorisation used by the proposal fail. A small
positive diagonal keeps `Σ` strictly positive definite. `noise_scaling` should
be far below the typical parameter variances (`1e-6`–`1e-4`) so the adaptation
is not biased.

Dispatch is on `Val{Bool}` so the disabled branch compiles away entirely.
"""
function add_noise!(rng::AbstractRNG, Σ, noise_scaling, n, ::Val{true})
    @inbounds for i in 1:n
        Σ[i, i] += noise_scaling * rand(rng)
    end
    return Σ
end

add_noise!(::AbstractRNG, Σ, _scaling, _n, ::Val{false}) = Σ

# Bool-valued fallback for the samplers that store the flag as a plain `Bool`
# rather than a `Val`.
function add_noise!(rng::AbstractRNG, Σ, noise_scaling, n, use::Bool)
    return add_noise!(rng, Σ, noise_scaling, n, use ? Val(true) : Val(false))
end

# ------------------------------------------------------------- transitions --

"""
    make_transition(state, accepted)

Build the transition for the current position of `state`, re-evaluating the
model to recover the constrained parameter values.
"""
function make_transition(state::AbstractAMGSState, accepted::Bool)
    return DynamicPPL.ParamsWithStats(state.θ, state.ldf, (; accepted=accepted))
end

"""
    reuse_transition(state, accepted)

Reuse the cached transition, rewriting only its `accepted` flag.

Valid whenever the position did not move, i.e. after a rejection. A well-tuned
random-walk chain rejects roughly 77% of the time, so this avoids the large
majority of the model re-evaluations that would otherwise be needed.
"""
function reuse_transition(state::AbstractAMGSState, accepted::Bool)
    old = state.transition
    return DynamicPPL.ParamsWithStats(old.params, merge(old.stats, (; accepted=accepted)))
end

# --------------------------------------------------------------- proposals --

"""
    propose!(rng, state, Σ, scale=1.0)

Draw a Gaussian random-walk proposal `θ' = θ + √scale · L z`, where `Σ = L Lᵀ`,
into the preallocated `state.θ_prop`, and return `(θ', log π(θ'))`.

The global scale is applied to the *factor* rather than to the covariance:
`chol(scale·Σ).L == √scale · chol(Σ).L` exactly, so folding it in here avoids
materialising the scaled `d×d` covariance as a temporary on every iteration.

The proposal is symmetric, so the Hastings correction cancels and the acceptance
ratio is just the log-density difference.

If `Σ` is not numerically positive definite the move is rejected by returning
`-Inf` rather than throwing, which keeps a chain alive through a transient bad
adaptation instead of aborting the run.
"""
function propose!(
    rng::AbstractRNG, state::AbstractAMGSState, Σ::AbstractMatrix, scale::Real=1.0
)
    θ_prop = state.θ_prop
    z = state.z_cache
    randn!(rng, z)

    chol = cholesky(Symmetric(Σ); check=false)
    if !issuccess(chol)
        copyto!(θ_prop, state.θ)
        return θ_prop, -Inf
    end

    # Fold the global scale into the factor: √scale · L, no scaled-Σ temporary.
    if scale != 1
        rmul!(z, sqrt(scale))
    end
    copyto!(θ_prop, z)
    lmul!(LowerTriangular(chol.L), θ_prop)
    @inbounds @simd for i in eachindex(θ_prop)
        θ_prop[i] += state.θ[i]
    end

    return θ_prop, LogDensityProblems.logdensity(state.ldf, θ_prop)
end

"""
    metropolis_move!(rng, state, Σ, scale=1.0)

One Metropolis accept/reject move from `state` using proposal covariance
`scale · Σ`, updating `state.θ` in place on acceptance. See [`propose!`](@ref)
for why the scale is kept separate from the matrix.

Returns `(accepted, logp, prob)`, where `logp` is the log-density at the
resulting position and `prob = min(1, exp(Δ))` is the realised acceptance
probability that drives the scale adaptation. Note `prob` is the *probability*,
not the 0/1 outcome: adapting on the probability gives a much lower-variance
signal than adapting on the indicator.
"""
function metropolis_move!(
    rng::AbstractRNG, state::AbstractAMGSState, Σ::AbstractMatrix, scale::Real=1.0
)
    θ_prop, lp_prop = propose!(rng, state, Σ, scale)
    lp_old = state.logp

    accepted = mh_accept_logdens(rng, lp_prop, lp_old)
    prob = exp(min(0.0, lp_prop - lp_old))

    if accepted
        copyto!(state.θ, θ_prop)
        return true, lp_prop, prob
    end
    return false, lp_old, prob
end

# ------------------------------------------------------------ Gibbs interface --

"""
    Turing.Inference.gibbs_get_raw_values(state::AbstractAMGSState)

Current parameter values in constrained ("raw") space, as the Turing Gibbs
machinery expects. Shared by every sampler in this package.
"""
function Turing.Inference.gibbs_get_raw_values(state::AbstractAMGSState)
    pws = DynamicPPL.ParamsWithStats(
        state.θ, state.ldf; include_log_probs=false, include_colon_eq=false
    )
    return pws.params
end

"""
    gibbs_rebuild(state, model, global_vals)

Shared first half of `gibbs_update_state!!`: rebuild the `LogDensityFunction`
against a model conditioned on the other Gibbs blocks' current values, copy the
refreshed parameters into `state.θ`, and recompute the cached log-density.

The `LogDensityFunction` *must* be rebuilt from the conditioned `model` rather
than reused from `state.ldf`, whose embedded model still holds the previous
blocks' values. Skipping this would compare log-densities under two different
conditionings in the next acceptance ratio, silently breaking the Gibbs sweep.

Returns `(new_ldf, θ, logp, transition)`. Each sampler then reassembles its own
state type around these, carrying its adaptation state over unchanged — the
proposal geometry for *this* block does not reset when other blocks move.
"""
function gibbs_rebuild(
    state::AbstractAMGSState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, new_params, _ = Turing.Inference.gibbs_recompute_ldf_and_params(
        state.ldf, model, global_vals
    )
    θ = copyto!(state.θ, new_params)
    logp = LogDensityProblems.logdensity(new_ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, new_ldf, (; accepted=true))
    return new_ldf, θ, logp, transition
end
