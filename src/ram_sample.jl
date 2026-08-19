"""
ram_sample.jl

Robust Adaptive Metropolis (RAM), Vihola (2012), https://arxiv.org/abs/1011.4381.

RAM differs structurally from the other samplers here. Instead of adapting a
covariance `Σ` and a separate global scale `exp(β)`, it adapts the Cholesky
factor `S` of the proposal covariance directly, with a single rank-one update
per iteration:

    S_{n+1} S_{n+1}ᵀ = S_n (I + η_n (α_n - α*) UUᵀ / ‖U‖²) S_nᵀ

where `U ~ N(0, I)` is the standard normal draw that generated the proposal
`θ' = θ + S_n U`, `α_n` is the realised acceptance probability and `α*` the
target. The shape and the scale are therefore learned together, and the scheme
provably drives the acceptance rate to `α*` without needing to estimate the
target's mean.

Because the update is rank one, it is applied with `lowrankupdate` /
`lowrankdowndate` on the Cholesky factor rather than by re-factorising, which is
`O(d²)` per iteration instead of `O(d³)`. Expanding the recursion shows why the
update vector is `√(η|α-α*|) · S U / ‖U‖`:

    S (I ± c UUᵀ/‖U‖²) Sᵀ = S Sᵀ ± c (S U)(S U)ᵀ / ‖U‖²

so a rank-one update with that vector reproduces the recursion exactly.
"""

"""
    RAM_Sampler(acceptance_rate, convergence_step; lb_eig=-Inf, ub_eig=Inf, bound_eigs=false)

Robust Adaptive Metropolis sampler.

# Fields
- `acceptance_rate::Real`: target acceptance rate `α*`.
- `convergence_step::Real`: decay exponent `γ` of the gain `η = (t+1)^(-γ)`.
- `lb_eig::Real`, `ub_eig::Real`: bounds on the eigenvalues of the proposal
  covariance, applied only when `bound_eigs` is true.
- `bound_eigs::Bool`: if true, an update that would push any eigenvalue of the
  proposal covariance outside `[lb_eig, ub_eig]` is rejected and the previous
  factor retained. This guards against the adaptation collapsing or exploding on
  badly conditioned targets, at the cost of an eigendecomposition per iteration.

# Example
```julia
spl = RAM_Sampler(0.234, 0.6)
spl = RAM_Sampler(0.234, 0.6; lb_eig=1e-8, ub_eig=1e8, bound_eigs=true)
```
"""
struct RAM_Sampler{T_accep<:Real,T_conv_step<:Real,T_lb_eig<:Real,T_ub_eig<:Real} <:
       AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    lb_eig::T_lb_eig
    ub_eig::T_ub_eig
    bound_eigs::Bool
end

function RAM_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
    lb_eig::Real=-Inf,
    ub_eig::Real=Inf,
    bound_eigs::Bool=false,
) where {T1<:Real,T2<:Real}
    lb_eig < ub_eig || throw(ArgumentError("lb_eig must be below ub_eig"))
    return RAM_Sampler(acceptance_rate, convergence_step, lb_eig, ub_eig, bound_eigs)
end

"""
    RAMState

State for [`RAM_Sampler`](@ref).

# Fields
- `ldf`: `LogDensityFunction` over the linked parameter vector.
- `transition`: cached transition for the current position.
- `θ`: current position in linked space.
- `logp`: cached log-density at `θ`.
- `kernel`: the `Cholesky` factorisation of the proposal covariance. Unlike the
  other samplers this *is* the adapted object — there is no separate `Σ` and no
  global scale `β`.
- `step`: iteration counter driving the gain `η`.
- `θ_prop`, `z_cache`: preallocated proposal and standard-normal buffers.
"""
struct RAMState{
    TLDF<:DynamicPPL.LogDensityFunction,
    TTrans<:DynamicPPL.ParamsWithStats,
    Tθ<:AbstractVector,
    TKernel<:Cholesky,
    TStep<:Integer,
    TL<:Real,
} <: AbstractAMGSState
    ldf::TLDF
    transition::TTrans
    θ::Tθ
    logp::TL
    kernel::TKernel
    step::TStep
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    init_RAMState(ldf, transition, θ, logp, kernel)

Build the initial RAM state, allocating the proposal buffers once.
"""
function init_RAMState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::Cholesky,
)
    return RAMState(ldf, transition, θ, logp, kernel, 1, similar(θ), similar(θ))
end

"""
    update_kernel(state, U, spl, prob)

Apply Vihola's rank-one update to the Cholesky factor and return the new factor.

`U` must be the standard normal draw that generated the accepted-or-rejected
proposal, and `prob` the realised acceptance probability. The sign of
`prob - α*` decides whether the factor is updated (proposal too conservative,
widen) or downdated (too aggressive, shrink).

A downdate can make the matrix numerically indefinite, in which case
`lowrankdowndate` throws; the previous factor is then retained, which keeps the
chain valid rather than aborting the run.
"""
function update_kernel(state::RAMState, U::AbstractVector, spl::RAM_Sampler, prob::Real)
    S = state.kernel
    nrm = norm(U)

    # Degenerate draw: no direction to update along.
    iszero(nrm) && return S

    η = (state.step + 1.0)^(-spl.convergence_step)
    accepdiff = prob - spl.acceptance_rate
    v = sqrt(η * abs(accepdiff)) .* (S.L * U) ./ nrm

    S_new = try
        if accepdiff >= 0
            LinearAlgebra.lowrankupdate(S, v)
        else
            LinearAlgebra.lowrankdowndate(S, v)
        end
    catch err
        # A downdate that would destroy positive definiteness: keep the old
        # factor rather than propagating a broken kernel.
        err isa LinearAlgebra.PosDefException || rethrow()
        return S
    end

    # Optionally reject updates that push the covariance's eigenvalues outside
    # the configured band. Note the eigenvalues wanted are those of the
    # covariance S*Sᵀ, not of the triangular factor.
    if spl.bound_eigs
        λ = eigvals(Symmetric(Matrix(S_new)))
        all(x -> spl.lb_eig <= x <= spl.ub_eig, λ) || return S
    end

    return S_new
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::RAM_Sampler) = true
Turing.Inference.init_strategy(::RAM_Sampler) = DynamicPPL.InitFromPrior()

"""
    propose_ram!(rng, state)

Draw `U ~ N(0, I)` and form the proposal `θ' = θ + S U` in the preallocated
buffers, returning `(θ', U, log π(θ'))`.

`U` is returned because the RAM adaptation is expressed in terms of the standard
normal draw, not the proposal increment.
"""
function propose_ram!(rng::AbstractRNG, state::RAMState)
    U = state.z_cache
    θ_prop = state.θ_prop
    randn!(rng, U)

    copyto!(θ_prop, U)
    lmul!(LowerTriangular(state.kernel.L), θ_prop)   # θ_prop = S U
    @inbounds @simd for i in eachindex(θ_prop)
        θ_prop[i] += state.θ[i]
    end

    return θ_prop, U, LogDensityProblems.logdensity(state.ldf, θ_prop)
end

function _ram_initial_state(rng::AbstractRNG, model::DynamicPPL.Model, initial_params)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    S = cholesky(Symmetric(initial_kernel(θ)))
    return init_RAMState(ldf, transition, θ, logp, S)
end

"""
    _ram_step!(rng, state, spl, discard_sample)

One RAM iteration: propose along the current Cholesky factor, accept or reject,
then apply the rank-one adaptation.
"""
function _ram_step!(
    rng::AbstractRNG, state::RAMState, spl::RAM_Sampler, discard_sample::Bool
)
    θ_prop, U, lp_prop = propose_ram!(rng, state)
    lp_old = state.logp

    accep = mh_accept_logdens(rng, lp_prop, lp_old)
    prob = exp(min(0.0, lp_prop - lp_old))
    logp = lp_old
    if accep
        copyto!(state.θ, θ_prop)
        logp = lp_prop
    end

    S = update_kernel(state, U, spl, prob)

    moved = RAMState(
        state.ldf,
        state.transition,
        state.θ,
        logp,
        S,
        state.step + 1,
        state.θ_prop,
        state.z_cache,
    )

    # Keep the transition cache consistent with the position regardless of
    # `discard_sample`, which only suppresses what is returned.
    transition = accep ? make_transition(moved, true) : reuse_transition(moved, false)

    new_state = RAMState(
        moved.ldf,
        transition,
        moved.θ,
        moved.logp,
        moved.kernel,
        moved.step,
        moved.θ_prop,
        moved.z_cache,
    )

    return (discard_sample ? nothing : transition), new_state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _ram_initial_state(rng, model, initial_params)
    return _ram_step!(rng, state, spl, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
    state::RAMState;
    discard_sample=false,
    kwargs...,
)
    return _ram_step!(rng, state, spl, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _ram_initial_state(rng, model, initial_params)
    return _ram_step!(rng, state, spl, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
    state::RAMState;
    discard_sample=false,
    kwargs...,
)
    return _ram_step!(rng, state, spl, discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::RAM_Sampler,
    state::RAMState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return RAMState(
        new_ldf,
        transition,
        θ,
        logp,
        state.kernel,
        state.step,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::RAM_Sampler, state) = missing
