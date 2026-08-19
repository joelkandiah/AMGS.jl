"""
amgs_sample.jl

Implementation of the Adaptive Metropolis-within-Gibbs (AMGS) sampler.

AMGS is a random-walk Metropolis sampler whose proposal covariance is adapted
online with a Robbins–Monro scheme (Haario et al. style covariance adaptation
combined with Andrieu & Thoms style global scale adaptation).

Because the proposal is a Gaussian random walk, the sampler requires an
**unconstrained, real-valued** parameter space. This is guaranteed structurally:
the sampler works exclusively on the *linked* (transformed) vector
representation of the model, obtained via a `DynamicPPL.LogDensityFunction`
built with `DynamicPPL.LinkAll()`. Constrained variables (e.g. variances,
simplices) are therefore transformed to ℝⁿ before proposing, and the log-density
used for the acceptance ratio is `getlogjoint_internal`, i.e. it *includes* the
log-Jacobian of that transformation. This is what makes the random walk a valid
MH kernel on the constrained model.

References
- Haario, Saksman & Tamminen (2001), "An adaptive Metropolis algorithm".
- Andrieu & Thoms (2008), "A tutorial on adaptive MCMC",
  https://link.springer.com/article/10.1007/s11222-008-9110-y
"""

# Start by defining the sampler

"""
    AMGS_Sampler(acceptance_rate, convergence_step)
    AMGS_Sampler(acceptance_rate, convergence_step, use_noise_stabiliser, noise_scaling)

Sampler type for the adaptive Metropolis-within-Gibbs algorithm.

# Fields
- `acceptance_rate::Real`: target acceptance rate for the Robbins–Monro update.
  `0.234` is the standard optimal value for random-walk MH in high dimension.
- `convergence_step::Real`: decay exponent `γ` of the adaptation gain
  `η = (t+1)^(-γ)`. Must satisfy `0.5 < γ ≤ 1` for the Robbins–Monro conditions
  (`Σηₜ = ∞`, `Σηₜ² < ∞`) to hold; `0.6` is a common choice.
- `use_noise_stabiliser::Val{Bool}`: compile-time flag to optionally add noise
  to the adapted covariance (see [`add_noise!`](@ref)).
- `noise_scaling::Real`: scale of the optional noise stabiliser.

# Example
```julia
spl = AMGS_Sampler(0.234, 0.6)
chain = sample(model, spl, 2_000; num_warmup = 1_000)
```
"""
struct AMGS_Sampler{T_accep<:Real,T_conv_step<:Real,T_scale<:Real,T_B} <:
       AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
end

function AMGS_Sampler(
    acceptance_rate::T1, convergence_step::T2
) where {T1<:Real,T2<:Real}
    return AMGS_Sampler(acceptance_rate, convergence_step, Val(false), 1e-6)
end

function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3,
) where {T1<:Real,T2<:Real,T3<:Real}
    return AMGS_Sampler(
        acceptance_rate, convergence_step, Val(use_noise_stabiliser), noise_scaling
    )
end

"""
    AMGSState

State container for the AMGS sampler.

All parameter vectors live in the **linked / unconstrained** space defined by
`ldf`.

# Fields
- `ldf`: `DynamicPPL.LogDensityFunction` over the linked parameter vector. Its
  log-density is `getlogjoint_internal` (log prior + log Jacobian + log
  likelihood), which is the correct MH target in transformed space.
- `transition`: cached `DynamicPPL.ParamsWithStats` for the current position,
  reused on rejection so that a rejected step costs no model re-evaluation.
- `θ`: current position in linked space.
- `logp`: cached `getlogjoint_internal` at `θ`.
- `μ`: running mean of `θ` used by the covariance adaptation.
- `kernel`: adapted proposal covariance `Σ` (before the `exp(β)` scaling).
- `step`: adaptation step counter, driving the Robbins–Monro gain.
- `β`: log of the global scale multiplier; the proposal covariance actually used
  is `exp(β) * kernel`.
- `Δ_cache`: preallocated scratch buffer (avoids per-step allocation).
- `θ_prop`: preallocated proposal buffer (avoids per-step allocation).
- `z_cache`: preallocated standard-normal draw buffer.
"""
struct AMGSState{
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
    init_AMGSState(ldf, transition, θ, logp, kernel)

Create the initial [`AMGSState`](@ref), allocating the running mean and the
scratch buffers once so that the per-step loop is allocation-free.

The running mean `μ` is initialised at the starting position `θ` (rather than at
zero) so the covariance adaptation is not biased by an arbitrary origin during
the first few iterations. `β` starts at zero, i.e. the initial global scale
multiplier `exp(β)` is 1.
"""
function init_AMGSState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
)
    step = 1
    β = zero(eltype(θ))
    return AMGSState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),          # μ initialised at the starting position
        kernel,
        step,
        β,
        similar(θ),       # Δ_cache
        similar(θ),       # θ_prop
        similar(θ),       # z_cache
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Robbins–Monro update of the running mean, the proposal covariance and the global
scale, performed in place on the buffers held by `state`.

With gain `η = (t+1)^(-γ)` (`γ = spl.convergence_step`) the updates are

    μ  ← μ + η (θ - μ)                       (running mean)
    Σ  ← Σ + η ((θ-μ)(θ-μ)ᵀ - Σ)             (running covariance, Haario et al.)
    β  ← β + η (α - α*)                      (global scale, Andrieu & Thoms)

where `α = prob` is the realised MH acceptance probability of the step just
taken and `α* = spl.acceptance_rate` is the target. The proposal covariance
actually used is `exp(β) Σ`, so `β` shrinks the proposal when the chain accepts
too rarely and inflates it when it accepts too often.

Note `Σ` is updated using `μ` *after* the mean update, and only the upper
triangle is computed before being mirrored, which halves the work and keeps `Σ`
exactly symmetric.

Returns the updated `(kernel, β)`; `μ` and `kernel` are mutated in place.
"""
function update_kernel!(
    rng::AbstractRNG, state::AMGSState, spl::AMGS_Sampler, prob::Real
)
    μ = state.μ
    Σ = state.kernel
    Δ = state.Δ_cache
    n = length(Δ)

    @fastmath begin
        # Robbins–Monro gain
        η = (state.step + 1.0)^(-spl.convergence_step)

        # Adapt the global scale towards the target acceptance rate
        β = state.β + η * (prob - spl.acceptance_rate)

        # Δ = θ - μ_t, taken against the mean *before* it is updated, then the
        # mean is advanced. This ordering is the one specified in the AMGS
        # pseudocode (Algorithm 1) and in Andrieu & Thoms Algorithm 4:
        #     Δ     = θ_{t+1} - μ_t
        #     μ_{t+1} = (1-η) μ_t + η θ_{t+1}   ==  μ_t + η Δ
        #     Σ_{t+1} = (1-η) Σ_t + η ΔΔᵀ       ==  Σ_t + η(ΔΔᵀ - Σ_t)
        #
        # Taking Δ against the *updated* mean instead would scale every residual
        # by (1-η), under-estimating the target covariance in the early
        # high-gain steps, and at η = 1 would collapse Σ to exactly zero.
        @inbounds @simd for i in 1:n
            Δ[i] = state.θ[i] - μ[i]
        end

        # Update the running mean: μ ← μ + ηΔ, i.e. (1-η)μ + ηθ.
        @inbounds @simd for i in 1:n
            μ[i] += η * Δ[i]
        end

        # Symmetric rank-one covariance update, in place
        @inbounds for i in 1:n
            Δi = Δ[i]
            for j in i:n
                Σ[i, j] += η * (Δi * Δ[j] - Σ[i, j])
                Σ[j, i] = Σ[i, j]  # preserve exact symmetry
            end
        end

        # Optional noise stabilisation (compiles away when disabled)
        add_noise!(rng, Σ, spl.noise_scaling, n, spl.use_noise_stabiliser)
    end

    return Σ, β
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_Sampler) = true

Turing.Inference.init_strategy(::AMGS_Sampler) = DynamicPPL.InitFromPrior()

"""
    _amgs_initial_step(rng, model, spl, initial_params)

Shared initialisation for the first `step` / `step_warmup` call: build the
linked log-density function, draw the starting point, and assemble the initial
state.
"""
function _amgs_initial_step(rng::AbstractRNG, model::DynamicPPL.Model, spl::AMGS_Sampler, initial_params)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    Σ = initial_kernel(θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    return init_AMGSState(ldf, transition, θ, logp, Σ)
end

####
#### AbstractMCMC interface
####

"""
    AbstractMCMC.step_warmup(rng, model, spl::AMGS_Sampler; kwargs...)

First warmup step: initialise the sampler and take one adapting move.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _amgs_initial_step(rng, model, spl, initial_params)
    return _amgs_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

"""
    AbstractMCMC.step_warmup(rng, model, spl::AMGS_Sampler, state; kwargs...)

Subsequent warmup step. Identical to [`AbstractMCMC.step`](@ref) — the AMGS
adaptation is run during both warmup and sampling, since the Robbins–Monro gain
decays on its own rather than being switched off.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler,
    state::AMGSState;
    discard_sample=false,
    kwargs...,
)
    return _amgs_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

"""
    AbstractMCMC.step(rng, model, spl::AMGS_Sampler; kwargs...)

First sampling step: initialise the sampler and take one adapting move.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _amgs_initial_step(rng, model, spl, initial_params)
    return _amgs_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

"""
    AbstractMCMC.step(rng, model, spl::AMGS_Sampler, state; kwargs...)

Subsequent sampling step: propose, accept/reject, then adapt.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler,
    state::AMGSState;
    discard_sample=false,
    kwargs...,
)
    return _amgs_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

"""
    _amgs_step!(rng, state, spl, Σ, discard_sample)

The single AMGS iteration shared by `step` and `step_warmup`: propose,
accept/reject, adapt the kernel, and emit the transition.
"""
function _amgs_step!(
    rng::AbstractRNG,
    state::AMGSState,
    spl::AMGS_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)

    # Adapt using the position the chain actually occupies after the move.
    _, β = update_kernel!(rng, state, spl, prob)

    # The cached transition must always be kept consistent with the current
    # position, even when the sample is discarded: `discard_sample` suppresses
    # what is *returned*, not the bookkeeping. Leaving the cache stale during
    # warmup would make the first post-warmup rejection emit the parameter
    # values from wherever the chain was when warmup began.
    transition = if accep
        make_transition(state, true)     # θ moved: must re-evaluate
    else
        reuse_transition(state, false)   # θ unchanged: reuse, no model eval
    end

    new_state = AMGSState(
        state.ldf,
        transition,
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

    return (discard_sample ? nothing : transition), new_state
end

####
#### Gibbs interface
####

"""
    Turing.Inference.gibbs_update_state!!(spl, state::AMGSState, model, global_vals)

Rebuild the state against a model conditioned on the other Gibbs blocks' current
values.

The `LogDensityFunction` must be rebuilt from the *conditioned* `model` (not
`state.ldf.model`, which holds stale conditioned values), and the cached
log-density must be recomputed, otherwise the next acceptance ratio would
compare densities under two different conditionings.

The adaptation state (`μ`, `kernel`, `β`, `step`) is deliberately carried over
unchanged: it describes the proposal geometry for *this* block, which does not
reset when the other blocks move.
"""
function Turing.Inference.gibbs_update_state!!(
    ::AMGS_Sampler,
    state::AMGSState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSState(
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

# Define function to get log evidence
getlogevidence(transitions, sampler::AMGS_Sampler, state) = missing
