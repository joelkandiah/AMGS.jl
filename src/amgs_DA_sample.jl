"""
amgs_DA_sample.jl

AMGS with **dual-averaging** adaptation of the global proposal scale.

The covariance `Σ` is adapted exactly as in base AMGS (Robbins–Monro / Haario),
but the global scale multiplier is tuned by Nesterov dual averaging — the same
scheme Stan and NUTS use for step-size adaptation (Hoffman & Gelman 2014,
Algorithm 6):

    Hₜ = α* - αₜ
    ḡₜ = (1 - 1/(t+t₀)) ḡₜ₋₁ + (1/(t+t₀)) Hₜ
    xₜ = μ - (√t / γ) ḡₜ
    x̄ₜ = t^(-κ) xₜ + (1 - t^(-κ)) x̄ₜ₋₁

with the proposal covariance then being `exp(x̄ₜ) Σ`.

Compared with the plain Robbins–Monro scale update in [`AMGS_Sampler`](@ref),
dual averaging is far less sensitive to the early, badly-scaled iterations: `ḡ`
averages the acceptance error over the whole history, and the `x̄` smoothing
damps the oscillation that a raw stochastic-approximation update shows when the
covariance is still moving.

This implementation uses the *averaged* iterate `x̄` as the live scale
throughout, rather than running on `x` and switching to `x̄` only at the end of
warmup. That is more conservative — the scale moves more smoothly — at the cost
of responding more slowly to a rapidly improving covariance.

Reference: Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR 15.
"""

"""
    AMGS_DA_Sampler(acceptance_rate, convergence_step)
    AMGS_DA_Sampler(acceptance_rate, convergence_step, use_noise_stabiliser, noise_scaling)

AMGS with dual-averaging adaptation of the global scale.

# Fields
- `acceptance_rate::Real`: target acceptance rate `α*`.
- `convergence_step::Real`: Robbins–Monro decay exponent `γ` for the *covariance*
  update (distinct from the dual-averaging `γ` below).
- `use_noise_stabiliser::Val{Bool}`, `noise_scaling::Real`: optional covariance
  noise stabiliser (see [`add_noise!`](@ref)).
- `μ::Real`: the point dual averaging shrinks the log-scale towards (default
  `0.0`, i.e. a scale multiplier of 1).
- `γ::Real`: dual-averaging gain; smaller means larger steps (default `0.05`).
- `t0::Real`: iteration offset damping early updates (default `10.0`).
- `κ::Real`: decay exponent of the `x̄` averaging, in `(0.5, 1]` (default `0.75`).

The `μ`, `γ`, `t0`, `κ` defaults are Stan's.

# Example
```julia
spl = AMGS_DA_Sampler(0.234, 0.6)
```
"""
struct AMGS_DA_Sampler{
    T_accep<:Real,
    T_conv_step<:Real,
    T_scale<:Real,
    T_B,
    T_μ<:Real,
    T_γ<:Real,
    T_t<:Real,
    T_κ<:Real,
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    μ::T_μ
    γ::T_γ
    t0::T_t
    κ::T_κ
end

function AMGS_DA_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool=false,
    noise_scaling::Real=1e-6;
    μ::Real=0.0,
    γ::Real=0.05,
    t0::Real=10.0,
    κ::Real=0.75,
) where {T1<:Real,T2<:Real}
    γ > 0 || throw(ArgumentError("dual-averaging γ must be positive"))
    t0 > 0 || throw(ArgumentError("dual-averaging t0 must be positive"))
    0.5 < κ <= 1 || throw(ArgumentError("dual-averaging κ must lie in (0.5, 1]"))
    return AMGS_DA_Sampler(
        acceptance_rate,
        convergence_step,
        Val(use_noise_stabiliser),
        noise_scaling,
        μ,
        γ,
        t0,
        κ,
    )
end

"""
    AMGSDAState

State for [`AMGS_DA_Sampler`](@ref).

Extends the base AMGS state with the two dual-averaging accumulators.

# Additional fields
- `x̄`: the averaged log-scale iterate; this is what `β` is set to and what
  actually scales the proposal.
- `ḡ`: the running average of the acceptance error `α* - α`.
"""
struct AMGSDAState{
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
    x̄::Tβ
    ḡ::Tβ
    Δ_cache::Tθ
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    init_AMGSDAState(ldf, transition, θ, logp, kernel)

Build the initial dual-averaging state. `x̄` and `ḡ` start at zero, so the
initial scale multiplier `exp(x̄)` is 1.

The step counter starts at 1, never 0: the dual-averaging smoothing weight
`t^(-κ)` is infinite at `t = 0`.
"""
function init_AMGSDAState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
)
    z = zero(eltype(θ))
    return AMGSDAState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),
        kernel,
        1,
        z,      # β
        z,      # x̄
        z,      # ḡ
        similar(θ),
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Adapt the covariance by Robbins–Monro and the global scale by dual averaging.

The covariance recursion is the usual Andrieu & Thoms Algorithm 4 form,

    Δ = θ - μ_t,  μ ← μ + ηΔ,  Σ ← Σ + η(ΔΔᵀ - Σ),

while the scale follows the Nesterov dual-averaging recursion described in the
file docstring. Returns `(Σ, x̄, ḡ)`; `μ` and `Σ` are mutated in place.
"""
function update_kernel!(
    rng::AbstractRNG, state::AMGSDAState, spl::AMGS_DA_Sampler, prob::Real
)
    μ = state.μ
    Σ = state.kernel
    Δ = state.Δ_cache
    n = length(Δ)
    step = state.step

    @fastmath begin
        # --- covariance: Robbins-Monro -----------------------------------
        η = (step + 1.0)^(-spl.convergence_step)

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

        # --- global scale: Nesterov dual averaging ------------------------
        Ht = spl.acceptance_rate - prob
        w = 1 / (step + spl.t0)
        ḡ = (1 - w) * state.ḡ + w * Ht
        x = spl.μ - (sqrt(step) / spl.γ) * ḡ
        decay = step^(-spl.κ)
        x̄ = decay * x + (1 - decay) * state.x̄
    end

    return Σ, x̄, ḡ
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_DA_Sampler) = true
Turing.Inference.init_strategy(::AMGS_DA_Sampler) = DynamicPPL.InitFromPrior()

function _da_initial_state(rng::AbstractRNG, model::DynamicPPL.Model, initial_params)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    return init_AMGSDAState(ldf, transition, θ, logp, initial_kernel(θ))
end

"""
    _da_step!(rng, state, spl, Σ, discard_sample)

One iteration: propose, accept/reject, then adapt the covariance and the
dual-averaged scale.
"""
function _da_step!(
    rng::AbstractRNG,
    state::AMGSDAState,
    spl::AMGS_DA_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)
    _, x̄, ḡ = update_kernel!(rng, state, spl, prob)

    moved = AMGSDAState(
        state.ldf,
        state.transition,
        state.θ,
        logp,
        state.μ,
        state.kernel,
        state.step + 1,
        x̄,          # β is the averaged dual-averaging iterate
        x̄,
        ḡ,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )

    # Keep the transition cache consistent with the position regardless of
    # `discard_sample`, which only suppresses what is returned.
    transition = accep ? make_transition(moved, true) : reuse_transition(moved, false)

    new_state = AMGSDAState(
        moved.ldf,
        transition,
        moved.θ,
        moved.logp,
        moved.μ,
        moved.kernel,
        moved.step,
        moved.β,
        moved.x̄,
        moved.ḡ,
        moved.Δ_cache,
        moved.θ_prop,
        moved.z_cache,
    )

    return (discard_sample ? nothing : transition), new_state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_DA_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _da_initial_state(rng, model, initial_params)
    return _da_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_DA_Sampler,
    state::AMGSDAState;
    discard_sample=false,
    kwargs...,
)
    return _da_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_DA_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _da_initial_state(rng, model, initial_params)
    return _da_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_DA_Sampler,
    state::AMGSDAState;
    discard_sample=false,
    kwargs...,
)
    return _da_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::AMGS_DA_Sampler,
    state::AMGSDAState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSDAState(
        new_ldf,
        transition,
        θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.β,
        state.x̄,
        state.ḡ,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::AMGS_DA_Sampler, state) = missing
