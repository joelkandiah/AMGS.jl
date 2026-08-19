"""
amgs_tempered_sample.jl

AMGS against a **tempered** posterior.

The sampler is the same adaptive random-walk Metropolis kernel as base AMGS; only
the target changes. Instead of the posterior it targets

    π_temp(θ) ∝ (p(θ) p(y|θ))^temp

transformed to the unconstrained space, i.e. `temp·(log prior + log lik) +
log|J|` — see [`TemperedLogJoint`](@ref).

`temp < 1` flattens the posterior, lowering the barriers between modes so the
chain mixes more freely; `temp > 1` sharpens it, concentrating draws near the
modes. `temp = 1` recovers base AMGS exactly.

!!! warning
    With `temp != 1` the chain's stationary distribution is the *tempered*
    posterior, not the posterior. Draws are only posterior samples at `temp = 1`;
    otherwise they need reweighting, or the sampler should be used as one rung of
    a tempering ladder.
"""

"""
    AMGS_Tempered_Sampler(acceptance_rate, convergence_step; temperature=1.0)
    AMGS_Tempered_Sampler(acceptance_rate, convergence_step, use_noise_stabiliser,
                          noise_scaling; temperature=1.0)

Tempered variant of [`AMGS_Sampler`](@ref).

# Fields
- `acceptance_rate::Real`: target acceptance rate for the scale adaptation.
- `convergence_step::Real`: Robbins–Monro decay exponent `γ`.
- `use_noise_stabiliser::Val{Bool}`, `noise_scaling::Real`: optional covariance
  noise stabiliser (see [`add_noise!`](@ref)).
- `temperature::Real`: the tempering exponent applied to the posterior.

# Example
```julia
spl = AMGS_Tempered_Sampler(0.234, 0.6; temperature = 0.5)
```
"""
struct AMGS_Tempered_Sampler{
    T_accep<:Real,T_conv_step<:Real,T_scale<:Real,T_B,T_Temp<:Real
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    temperature::T_Temp
end

function AMGS_Tempered_Sampler(
    acceptance_rate::T1, convergence_step::T2; temperature::Real=1.0
) where {T1<:Real,T2<:Real}
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    return AMGS_Tempered_Sampler(
        acceptance_rate, convergence_step, Val(false), 1e-6, temperature
    )
end

function AMGS_Tempered_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3;
    temperature::Real=1.0,
) where {T1<:Real,T2<:Real,T3<:Real}
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    return AMGS_Tempered_Sampler(
        acceptance_rate,
        convergence_step,
        Val(use_noise_stabiliser),
        noise_scaling,
        temperature,
    )
end

"""
    AMGSTemperedState

State for [`AMGS_Tempered_Sampler`](@ref).

Identical to [`AMGSState`](@ref) except that `ldf` targets the tempered
posterior and `logp` caches the *tempered* log-density — that is what drives the
acceptance ratio. The untempered log prior, log likelihood and log joint are
still reported on each transition (they come free from the accumulators), so the
untempered posterior density remains available for reweighting or diagnostics.
"""
struct AMGSTemperedState{
    TLDF<:DynamicPPL.LogDensityFunction,
    TTrans<:DynamicPPL.ParamsWithStats,
    Tθ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tβ<:Real,
    TL<:Real,
} <: AbstractAMGSState
    ldf::TLDF
    transition::TTrans
    θ::Tθ
    logp::TL
    μ::Tθ
    kernel::TKernel
    step::TStep
    β::Tβ
    Δ_cache::Tθ
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    init_AMGSTemperedState(ldf, transition, θ, logp, kernel)

Build the initial tempered state, allocating every buffer once.
"""
function init_AMGSTemperedState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
)
    return AMGSTemperedState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),
        kernel,
        1,
        zero(eltype(θ)),
        similar(θ),
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Robbins–Monro adaptation, identical to base AMGS: the tempering changes the
target, not the adaptation rule.

    Δ = θ - μ_t,  μ ← μ + ηΔ,  Σ ← Σ + η(ΔΔᵀ - Σ),  β ← β + η(α - α*)

Returns `(Σ, β)`; `μ` and `Σ` are mutated in place.
"""
function update_kernel!(
    rng::AbstractRNG, state::AMGSTemperedState, spl::AMGS_Tempered_Sampler, prob::Real
)
    μ = state.μ
    Σ = state.kernel
    Δ = state.Δ_cache
    n = length(Δ)

    @fastmath begin
        η = (state.step + 1.0)^(-spl.convergence_step)
        β = state.β + η * (prob - spl.acceptance_rate)

        # Residual against the mean *before* it is updated, then advance the
        # mean -- the ordering specified in the AMGS pseudocode (Algorithm 1)
        # and Andrieu & Thoms Algorithm 4. See `update_kernel!` in
        # amgs_sample.jl for the full recursion.
        @inbounds @simd for i in 1:n
            Δ[i] = state.θ[i] - μ[i]
        end
        @inbounds @simd for i in 1:n
            μ[i] += η * Δ[i]
        end
        @inbounds for i in 1:n
            Δi = Δ[i]
            for j in i:n
                Σ[i, j] += η * (Δi * Δ[j] - Σ[i, j])
                Σ[j, i] = Σ[i, j]
            end
        end
        add_noise!(rng, Σ, spl.noise_scaling, n, spl.use_noise_stabiliser)
    end

    return Σ, β
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_Tempered_Sampler) = true
Turing.Inference.init_strategy(::AMGS_Tempered_Sampler) = DynamicPPL.InitFromPrior()

"Transition carrying the tempered log-density alongside the untempered stats."
function _tempered_transition(state::AMGSTemperedState, accepted::Bool)
    return DynamicPPL.ParamsWithStats(
        state.θ, state.ldf, (; accepted=accepted, lp_tempered=state.logp)
    )
end

function _tempered_reuse(state::AMGSTemperedState, accepted::Bool)
    old = state.transition
    return DynamicPPL.ParamsWithStats(
        old.params, merge(old.stats, (; accepted=accepted, lp_tempered=state.logp))
    )
end

function _tempered_initial_state(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::AMGS_Tempered_Sampler, initial_params
)
    ldf = build_ldf(model, rng, initial_params, TemperedLogJoint(spl.temperature))
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(
        θ, ldf, (; accepted=true, lp_tempered=logp)
    )
    return init_AMGSTemperedState(ldf, transition, θ, logp, initial_kernel(θ))
end

"""
    _tempered_step!(rng, state, spl, Σ, discard_sample)

One tempered AMGS iteration: propose, accept/reject against the tempered target,
then adapt.
"""
function _tempered_step!(
    rng::AbstractRNG,
    state::AMGSTemperedState,
    spl::AMGS_Tempered_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)
    _, β = update_kernel!(rng, state, spl, prob)

    moved = AMGSTemperedState(
        state.ldf,
        state.transition,
        state.θ,
        logp,
        state.μ,
        state.kernel,
        state.step + 1,
        β,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )

    # Always keep the cached transition consistent with the current position;
    # `discard_sample` suppresses the *return*, not the bookkeeping. A stale
    # cache would make the first post-warmup rejection report the parameters
    # from the start of warmup alongside the current log-density.
    transition = if accep
        _tempered_transition(moved, true)     # θ moved: must re-evaluate
    else
        _tempered_reuse(moved, false)         # θ unchanged: reuse
    end

    new_state = AMGSTemperedState(
        moved.ldf,
        transition,
        moved.θ,
        moved.logp,
        moved.μ,
        moved.kernel,
        moved.step,
        moved.β,
        moved.Δ_cache,
        moved.θ_prop,
        moved.z_cache,
    )

    return (discard_sample ? nothing : transition), new_state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Tempered_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _tempered_initial_state(rng, model, spl, initial_params)
    return _tempered_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Tempered_Sampler,
    state::AMGSTemperedState;
    discard_sample=false,
    kwargs...,
)
    return _tempered_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Tempered_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _tempered_initial_state(rng, model, spl, initial_params)
    return _tempered_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Tempered_Sampler,
    state::AMGSTemperedState;
    discard_sample=false,
    kwargs...,
)
    return _tempered_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::AMGS_Tempered_Sampler,
    state::AMGSTemperedState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSTemperedState(
        new_ldf,
        transition,
        θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.β,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::AMGS_Tempered_Sampler, state) = missing
