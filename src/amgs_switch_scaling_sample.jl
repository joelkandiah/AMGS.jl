"""
amgs_switch_scaling_sample.jl

AMGS with a **switchable global-scaling adaptation** rule.

The covariance `Σ` is always adapted by the Robbins–Monro / Haario recursion, as
in base AMGS. What varies is how the scalar log-scale `β` — the multiplier in
the proposal covariance `exp(β) Σ` — is driven towards the target acceptance
rate. The rule is selected by the `method` symbol and dispatched on `Val{method}`
so the choice is resolved at compile time and costs nothing per iteration.

Available methods (all seek `α = α*`, writing `e = α - α*` and
`γₜ = (t+1)^(-convergence_step)`):

| `method`  | Update of `β`                                            |
|:----------|:---------------------------------------------------------|
| `:RM`     | `β += γₜ e` — plain Robbins–Monro stochastic approximation |
| `:DA`     | Nesterov dual averaging (Hoffman & Gelman 2014)           |
| `:EMA`    | exponential moving average towards `β + e`                |
| `:LOG`    | Robbins–Monro in `u`, reported as `β = log σ(u) < 0`       |
| `:NR`     | Newton–Raphson: `β += γₜ e / σ²`, `σ² = α(1-α)`            |
| `:SINR`   | scale-invariant: `β += γₜ e / (√v̂ + ε)`                   |
| `:CSINR`  | as `:SINR` but with the denominator floored at `σ²`       |

The Newton–Raphson family divides the error by an estimate of its own scale,
which makes the step size insensitive to how informative each acceptance
decision is: `σ² = α(1-α)` is the Bernoulli variance of a single accept/reject,
so `e/σ²` is the natural Newton step for the root-finding problem `E[α] = α*`.

!!! note "`:EMA` is algebraically identical to `:RM`"
    `β ← (1-γₜ)β + γₜ(β + e)` expands to `β + γₜ e`, exactly the `:RM` update.
    It is kept as a distinct method because it is named separately in the
    original design, but it provides no different behaviour.

!!! warning "`:LOG` can only shrink the proposal"
    `:LOG` sets `β = log(sigmoid(·))`, which is strictly negative, so
    `exp(β) < 1` always: the scale multiplier can never exceed one. That is only
    appropriate when the initial covariance is known to be too wide.
"""

"""
    AMGS_SGS_Sampler(acceptance_rate, convergence_step; method=:DA, ...)
    AMGS_SGS_Sampler(acceptance_rate, convergence_step, use_noise_stabiliser,
                     noise_scaling; method=:DA, ...)

AMGS with a selectable global-scaling adaptation rule.

# Fields
- `acceptance_rate::Real`: target acceptance rate `α*`.
- `convergence_step::Real`: Robbins–Monro decay exponent, used for both the
  covariance update and the `γₜ` gain of the scaling rules.
- `use_noise_stabiliser::Val{Bool}`, `noise_scaling::Real`: optional covariance
  noise stabiliser (see [`add_noise!`](@ref)).
- `μ`, `γ`, `t0`, `κ`: dual-averaging parameters (`:DA`); `γ` doubles as the
  exponential-moving-average rate for the variance estimate `v̂` used by `:SINR`
  and `:CSINR`.
- `method::Val{Symbol}`: which scaling rule to use.

# Example
```julia
spl = AMGS_SGS_Sampler(0.234, 0.6; method = :NR)
```
"""
struct AMGS_SGS_Sampler{
    T_accep<:Real,
    T_conv_step<:Real,
    T_scale<:Real,
    T_B,
    T_μ<:Real,
    T_γ<:Real,
    T_t<:Real,
    T_κ<:Real,
    M,
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    μ::T_μ
    γ::T_γ
    t0::T_t
    κ::T_κ
    method::Val{M}
end

"The scaling rules this sampler understands."
const SGS_METHODS = (:RM, :DA, :EMA, :LOG, :NR, :SINR, :CSINR)

function AMGS_SGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool=false,
    noise_scaling::Real=1e-6;
    method::Symbol=:DA,
    μ::Real=0.0,
    γ::Real=0.05,
    t0::Real=10.0,
    κ::Real=0.75,
) where {T1<:Real,T2<:Real}
    method in SGS_METHODS ||
        throw(ArgumentError("unknown method $method; expected one of $(SGS_METHODS)"))
    γ > 0 || throw(ArgumentError("γ must be positive"))
    t0 > 0 || throw(ArgumentError("t0 must be positive"))
    0.5 < κ <= 1 || throw(ArgumentError("κ must lie in (0.5, 1]"))
    return AMGS_SGS_Sampler(
        acceptance_rate,
        convergence_step,
        Val(use_noise_stabiliser),
        noise_scaling,
        μ,
        γ,
        t0,
        κ,
        Val(method),
    )
end

####
#### Global-scaling adaptation states
####

"""
    GlobalScalingAdaptationState{T}

Supertype of the per-method state carried by the global-scaling rules. Every
subtype stores at least the current log-scale `β`; methods that need extra
accumulators (dual averaging, the scale-invariant Newton rules) add them.
"""
abstract type GlobalScalingAdaptationState{T<:Real} end

"Plain Robbins–Monro: only the log-scale is carried."
struct RobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

"Dual averaging: the averaged iterate `β` plus the averaged error `ḡ`."
struct DualAveragingState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
    ḡ::T
end

"Exponential moving average (algebraically the same as `RobbinsMonroState`)."
struct ExponentialMovingAverageState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

"""
Robbins–Monro accumulated in an unconstrained variable `u`, with the reported
log-scale `β = log(sigmoid(u))` (always negative, so `exp(β) < 1`).

Keeping `u` separate from `β` is essential: feeding `β` back into its own
Robbins–Monro update composes `log∘sigmoid` with itself once per iteration, and
that map has no fixed point — it drives the scale to zero regardless of the
acceptance rate.
"""
struct LogisticRobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
    u::T
end

"Newton–Raphson Robbins–Monro, dividing the error by the Bernoulli variance."
struct NewtonRaphsonRobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

"Scale-invariant Newton–Raphson: carries the running error variance `v̂`."
struct ScaleInvariantNewtonRaphsonRobbinsMonroState{T<:Real} <:
       GlobalScalingAdaptationState{T}
    β::T
    v̂::T
end

"The current log-scale of any adaptation state."
get_beta(s::GlobalScalingAdaptationState) = s.β

init_global_adaptation_state(::Val{:RM}, ::Type{T}) where {T<:Real} =
    RobbinsMonroState{T}(zero(T))
init_global_adaptation_state(::Val{:DA}, ::Type{T}) where {T<:Real} =
    DualAveragingState{T}(zero(T), zero(T))
init_global_adaptation_state(::Val{:EMA}, ::Type{T}) where {T<:Real} =
    ExponentialMovingAverageState{T}(zero(T))
init_global_adaptation_state(::Val{:LOG}, ::Type{T}) where {T<:Real} =
    LogisticRobbinsMonroState{T}(T(log(one(T) / 2)), zero(T))
init_global_adaptation_state(::Val{:NR}, ::Type{T}) where {T<:Real} =
    NewtonRaphsonRobbinsMonroState{T}(zero(T))
init_global_adaptation_state(::Val{:SINR}, ::Type{T}) where {T<:Real} =
    ScaleInvariantNewtonRaphsonRobbinsMonroState{T}(zero(T), one(T))
init_global_adaptation_state(::Val{:CSINR}, ::Type{T}) where {T<:Real} =
    ScaleInvariantNewtonRaphsonRobbinsMonroState{T}(zero(T), one(T))

####
#### The scaling rules
####

"""
    adapt_scale(method, spl, method_state, step, prob)

Advance the global log-scale by one iteration under the selected rule, returning
a new adaptation state. `prob` is the realised MH acceptance probability of the
step just taken.
"""
function adapt_scale end

function adapt_scale(::Val{:RM}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.convergence_step)
    newβ = method_state.β + γt * (prob - spl.acceptance_rate)
    return RobbinsMonroState{typeof(newβ)}(newβ)
end

# Nesterov dual averaging, Hoffman & Gelman (2014), JMLR 15, Algorithm 6.
function adapt_scale(::Val{:DA}, spl, method_state, step, prob)
    Ht = spl.acceptance_rate - prob
    w = 1 / (step + spl.t0)
    ḡ = (1 - w) * method_state.ḡ + w * Ht
    x = spl.μ - ((sqrt(step) / spl.γ) * ḡ)
    decay = step^(-spl.κ)
    x̄ = decay * x + (1 - decay) * method_state.β
    return DualAveragingState{typeof(x̄)}(x̄, ḡ)
end

# Exponential moving average towards `β + e`. Expanding the convex combination
# gives β + η(prob - α*), i.e. exactly the `:RM` update -- see the module note.
function adapt_scale(::Val{:EMA}, spl, method_state, step, prob)
    η = (step + 1.0)^(-spl.convergence_step)
    target = method_state.β + (prob - spl.acceptance_rate)
    newβ = (1 - η) * method_state.β + η * target
    return ExponentialMovingAverageState{typeof(newβ)}(newβ)
end

# Robbins-Monro accumulated in the unconstrained `u`, then squashed to a
# bounded log-scale. The result is always negative, so this rule can only ever
# shrink the proposal relative to `Σ`.
#
# NOTE: the original applied `log∘sigmoid` to the accumulator itself, so each
# iteration re-squashed an already-squashed value. That composition has no fixed
# point -- it drives β to -Inf and the chain grinds to a halt at acceptance 1.0
# regardless of the target. Accumulating in `u` and squashing only the reported
# value preserves the intended bounded parameterisation.
function adapt_scale(::Val{:LOG}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.convergence_step)
    u = method_state.u + γt * (prob - spl.acceptance_rate)
    newβ = -log1p(exp(-u))              # == log(sigmoid(u)), numerically stable
    return LogisticRobbinsMonroState{typeof(newβ)}(newβ, u)
end

# Newton-Raphson: divide the acceptance error by the Bernoulli variance of a
# single accept/reject decision, α(1-α).
#
# NOTE: the variance is floored at α*(1-α*), the Bernoulli variance *at the
# target* -- i.e. the curvature at the root the method is seeking. The original
# floored it at 1e-4, which is degenerate: whenever a proposal is certain to be
# accepted `prob` is exactly 1.0, so α(1-α) is exactly 0, the floor engages, and
# the update takes a step of order γₜ·e/1e-4 ≈ 5000 in log-scale. That overflows
# `exp(β)` to `Inf` on the very first iterations and the sampler returns NaN.
function adapt_scale(::Val{:NR}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.convergence_step)
    floor_σ2 = spl.acceptance_rate * (1 - spl.acceptance_rate)
    σ2 = clamp(abs(prob * (1 - prob)), floor_σ2, 0.25)
    newβ = method_state.β + γt * ((prob - spl.acceptance_rate) / σ2)
    return NewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ)
end

# Scale-invariant Newton-Raphson: normalise the step by a running estimate `v̂`
# of the squared acceptance error, in the manner of RMSProp/Adam.
#
# NOTE: `v̂` accumulates `(prob - acceptance_rate)^2`. The original code squared
# `(prob - convergence_step)` here, which compared an acceptance probability
# against the Robbins-Monro *decay exponent* -- two unrelated quantities. With
# the usual settings (α* = 0.234, convergence_step = 0.6) that inflated `v̂` and
# systematically damped the update.
function adapt_scale(::Val{:SINR}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.convergence_step)
    err = prob - spl.acceptance_rate
    v̂ = (1 - spl.γ) * method_state.v̂ + spl.γ * err^2
    newβ = method_state.β + γt * (err / (sqrt(v̂) + 1e-6))
    return ScaleInvariantNewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ, v̂)
end

# As `:SINR`, but the denominator is floored at the Bernoulli variance so the
# step cannot blow up when the error estimate is momentarily tiny.
function adapt_scale(::Val{:CSINR}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.convergence_step)
    σ2 = clamp(prob * (1 - prob), spl.acceptance_rate * (1 - spl.acceptance_rate), 0.25)
    err = prob - spl.acceptance_rate
    v̂ = (1 - spl.γ) * method_state.v̂ + spl.γ * err^2
    denom = max(sqrt(v̂), σ2)
    newβ = method_state.β + γt * (err / denom)
    return ScaleInvariantNewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ, v̂)
end

####
#### State
####

"""
    AMGSSGSState

State for [`AMGS_SGS_Sampler`](@ref).

Identical to [`AMGSState`](@ref) except that the scalar log-scale lives inside
`global_scale_adaptation_state`, whose concrete type depends on the selected
method. `β` mirrors that state's current log-scale so the generic helpers can
read it uniformly.
"""
struct AMGSSGSState{
    TLDF<:DynamicPPL.LogDensityFunction,
    TTrans<:DynamicPPL.ParamsWithStats,
    Tθ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tβ<:Real,
    TL<:Real,
    T_GSAS<:GlobalScalingAdaptationState,
} <: AbstractAMGSState
    ldf::TLDF
    transition::TTrans
    θ::Tθ
    logp::TL
    μ::Tθ
    kernel::TKernel
    step::TStep
    β::Tβ
    global_scale_adaptation_state::T_GSAS
    Δ_cache::Tθ
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    init_AMGSSGSState(ldf, transition, θ, logp, kernel, spl)

Build the initial switch-scaling state, including the method-specific adaptation
state. The step counter starts at 1, never 0, because the dual-averaging
smoothing weight `t^(-κ)` is infinite at `t = 0`.
"""
function init_AMGSSGSState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
    spl::AMGS_SGS_Sampler,
)
    method_state = init_global_adaptation_state(spl.method, eltype(θ))
    return AMGSSGSState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),
        kernel,
        1,
        get_beta(method_state),
        method_state,
        similar(θ),
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Adapt the covariance by Robbins–Monro (Andrieu & Thoms Algorithm 4) and the
global log-scale by the selected rule.

Returns `(Σ, method_state)`; `μ` and `Σ` are mutated in place.
"""
function update_kernel!(
    rng::AbstractRNG, state::AMGSSGSState, spl::AMGS_SGS_Sampler, prob::Real
)
    μ = state.μ
    Σ = state.kernel
    Δ = state.Δ_cache
    n = length(Δ)

    @fastmath begin
        η = (state.step + 1.0)^(-spl.convergence_step)

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

    method_state = adapt_scale(
        spl.method, spl, state.global_scale_adaptation_state, state.step, prob
    )
    return Σ, method_state
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_SGS_Sampler) = true
Turing.Inference.init_strategy(::AMGS_SGS_Sampler) = DynamicPPL.InitFromPrior()

function _sgs_initial_state(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::AMGS_SGS_Sampler, initial_params
)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    return init_AMGSSGSState(ldf, transition, θ, logp, initial_kernel(θ), spl)
end

"""
    _sgs_step!(rng, state, spl, Σ, discard_sample)

One iteration: propose, accept/reject, then adapt the covariance and the global
scale under the selected rule.
"""
function _sgs_step!(
    rng::AbstractRNG,
    state::AMGSSGSState,
    spl::AMGS_SGS_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)
    _, method_state = update_kernel!(rng, state, spl, prob)

    moved = AMGSSGSState(
        state.ldf,
        state.transition,
        state.θ,
        logp,
        state.μ,
        state.kernel,
        state.step + 1,
        get_beta(method_state),
        method_state,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )

    # Keep the transition cache consistent with the position regardless of
    # `discard_sample`, which only suppresses what is returned.
    transition = accep ? make_transition(moved, true) : reuse_transition(moved, false)

    new_state = AMGSSGSState(
        moved.ldf,
        transition,
        moved.θ,
        moved.logp,
        moved.μ,
        moved.kernel,
        moved.step,
        moved.β,
        moved.global_scale_adaptation_state,
        moved.Δ_cache,
        moved.θ_prop,
        moved.z_cache,
    )

    return (discard_sample ? nothing : transition), new_state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_SGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _sgs_initial_state(rng, model, spl, initial_params)
    return _sgs_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_SGS_Sampler,
    state::AMGSSGSState;
    discard_sample=false,
    kwargs...,
)
    return _sgs_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_SGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _sgs_initial_state(rng, model, spl, initial_params)
    return _sgs_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_SGS_Sampler,
    state::AMGSSGSState;
    discard_sample=false,
    kwargs...,
)
    return _sgs_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::AMGS_SGS_Sampler,
    state::AMGSSGSState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSSGSState(
        new_ldf,
        transition,
        θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.β,
        state.global_scale_adaptation_state,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::AMGS_SGS_Sampler, state) = missing
