"""
amgs_stopping_sample.jl

AMGS with a **stopping rule** for the adaptation.

This is Algorithm 4 of Andrieu & Thoms (2008) combined with the stopping
extension in the appendix of https://arxiv.org/abs/2408.04178.

Ordinary AMGS adapts forever, with the Robbins–Monro gain decaying as
`(t+1)^-γ`. That is asymptotically valid but means the chain is never exactly
Markov. This variant instead *freezes* the proposal once it is judged good
enough: the realised acceptance rate is averaged over blocks of
`average_interval` iterations, and as soon as that block average falls inside
`(lower_stopping_accep, upper_stopping_accep)` — or `max_adapt_period` is
exhausted — adaptation stops for good and the kernel becomes a fixed,
genuinely Markov random-walk Metropolis kernel.

Note the adaptation is applied *per block* rather than per iteration: the
Robbins–Monro update fires only at block boundaries, which is what makes the
block-averaged acceptance rate a meaningful stopping statistic.
"""

"""
    AMGS_Stopping_Sampler(acceptance_rate, convergence_step, max_adapt_period,
                          lower_stopping_accep, upper_stopping_accep, average_interval)

AMGS sampler that stops adapting once its acceptance rate settles.

# Fields
- `acceptance_rate::Real`: target acceptance rate for the scale adaptation.
- `convergence_step::Real`: Robbins–Monro decay exponent `γ`.
- `max_adapt_period::Integer`: hard cap on adaptation steps; `-1` disables the
  cap and relies solely on the acceptance-rate criterion.
- `lower_stopping_accep::Real`, `upper_stopping_accep::Real`: the acceptance-rate
  window that, once the block average lands inside it, freezes the adaptation.
- `average_interval::Integer`: block length over which the acceptance rate is
  averaged, and the period at which the kernel is updated.
- `use_noise_stabiliser::Bool`, `noise_scaling::Real`: optional covariance
  noise stabiliser (see [`add_noise!`](@ref)).

# Example
```julia
spl = AMGS_Stopping_Sampler(0.234, 0.6, 10_000, 0.15, 0.35, 100)
```
"""
struct AMGS_Stopping_Sampler{
    T_accep<:Real,
    T_conv_step<:Real,
    T_adapt_period<:Integer,
    T_accep_l<:Real,
    T_accep_u<:Real,
    T_accep_interval<:Integer,
    T_scale<:Real,
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    max_adapt_period::T_adapt_period
    lower_stopping_accep::T_accep_l
    upper_stopping_accep::T_accep_u
    average_interval::T_accep_interval
    use_noise_stabiliser::Bool
    noise_scaling::T_scale
end

function AMGS_Stopping_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    max_adapt_period::T3,
    lower_stopping_accep::T4,
    upper_stopping_accep::T5,
    average_interval::T6,
) where {T1<:Real,T2<:Real,T3<:Integer,T4<:Real,T5<:Real,T6<:Integer}
    average_interval > 0 || throw(ArgumentError("average_interval must be positive"))
    lower_stopping_accep < upper_stopping_accep ||
        throw(ArgumentError("lower_stopping_accep must be below upper_stopping_accep"))
    return AMGS_Stopping_Sampler{T1,T2,T3,T4,T5,T6,Float64}(
        acceptance_rate,
        convergence_step,
        max_adapt_period,
        lower_stopping_accep,
        upper_stopping_accep,
        average_interval,
        false,
        1e-6,
    )
end

"""
    AMGSStoppingState

State for [`AMGS_Stopping_Sampler`](@ref).

Extends the base AMGS state with the bookkeeping needed by the stopping rule.

# Additional fields
- `stopped::Bool`: once true, the proposal is frozen and no further adaptation
  occurs.
- `mean_accep`: acceptance rate accumulated over the current block; reset at
  each kernel update.
- `μ_running_avg`, `kernel_running_calc`: within-block running mean and
  covariance maintained by Welford's online algorithm.

!!! note
    `μ_running_avg` / `kernel_running_calc` are maintained as block diagnostics
    but are *not* consumed by [`update_kernel!`](@ref), which adapts from the
    instantaneous position, matching the original implementation.
"""
struct AMGSStoppingState{
    TLDF<:DynamicPPL.LogDensityFunction,
    TTrans<:DynamicPPL.ParamsWithStats,
    Tθ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tβ<:Real,
    TL<:Real,
    TAccep<:Real,
} <: AbstractAMGSState
    ldf::TLDF
    transition::TTrans
    θ::Tθ
    logp::TL
    μ::Tθ
    kernel::TKernel
    step::TStep
    β::Tβ
    stopped::Bool
    mean_accep::TAccep
    μ_running_avg::Tθ
    kernel_running_calc::TKernel
    Δ_cache::Tθ
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    init_AMGSStoppingState(ldf, transition, θ, logp, kernel)

Build the initial stopping state, allocating every buffer once.
"""
function init_AMGSStoppingState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
)
    return AMGSStoppingState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),            # μ
        kernel,
        1,                  # step (never 0: η = (step+1)^-γ must stay below 1)
        zero(eltype(θ)),    # β
        false,              # stopped
        0.0,                # mean_accep
        copy(θ),            # μ_running_avg (own storage; must not alias μ)
        zero(kernel),       # kernel_running_calc
        similar(θ),
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Robbins–Monro update of mean, covariance and global scale — identical in form to
the base AMGS update, but fired once per `average_interval` block rather than
every iteration.

    μ ← μ + η(θ - μ),  Σ ← Σ + η((θ-μ)(θ-μ)ᵀ - Σ),  β ← β + η(α - α*)

Returns `(Σ, β)`; `μ` and `Σ` are mutated in place.
"""
function update_kernel!(
    rng::AbstractRNG, state::AMGSStoppingState, spl::AMGS_Stopping_Sampler, prob::Real
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

"""
    check_end_adaptation(state, spl)

Decide whether adaptation should stop, given the acceptance rate averaged over
the block that has just completed. Called only at block boundaries.

Stops when either the hard `max_adapt_period` cap is reached, or the block
average acceptance rate lies strictly inside the target window.
"""
function check_end_adaptation(state::AMGSStoppingState, spl::AMGS_Stopping_Sampler)
    # Hard cap on the adaptation period (-1 disables it).
    if (spl.max_adapt_period != -1) && (state.step >= spl.max_adapt_period)
        return true
    end
    # Acceptance rate has settled inside the target window.
    if (state.mean_accep > spl.lower_stopping_accep) &&
        (state.mean_accep < spl.upper_stopping_accep)
        return true
    end
    return false
end

"""
    block_update_state(state, accepted, spl)

Advance the block bookkeeping by one iteration.

`average_interval` is an *averaging* window for the acceptance rate, not an
adaptation period: the kernel is adapted every iteration (as in base AMGS), and
this function separately accumulates the acceptance rate and the within-block
running mean/covariance. At the end of each block the accumulated acceptance
rate is tested against the stopping window and then reset.

Keeping the reset here — rather than inside [`update_kernel!`](@ref) — is what
makes the stopping rule work at all. If the accumulator were cleared on every
kernel update it would never rise above `1/average_interval`, so the
acceptance-rate criterion could never fire and only `max_adapt_period` could
ever stop the adaptation.

Once `stopped` is set the state is returned untouched: the kernel is frozen.
"""
function block_update_state(
    state::AMGSStoppingState, accepted::Bool, spl::AMGS_Stopping_Sampler
)
    if state.stopped
        # Still advance the step counter so the chain length stays meaningful.
        return _with_step(state, state.step + 1)
    end

    step = state.step + 1
    θ = state.θ
    μ_run = state.μ_running_avg
    Σ_run = state.kernel_running_calc

    # 0-based position of this iteration within the current block.
    step_mod = (step - 1) % spl.average_interval

    if step_mod == 0
        # First iteration of a new block: seed the running mean.
        copyto!(μ_run, θ)
        fill!(Σ_run, zero(eltype(Σ_run)))
    else
        # Welford online update, in place.
        n = step_mod + 1
        Δ = state.Δ_cache
        @inbounds @simd for i in eachindex(θ)
            Δ[i] = θ[i] - μ_run[i]
            μ_run[i] += Δ[i] / n
        end
        @inbounds for i in eachindex(θ)
            for j in eachindex(θ)
                Σ_run[i, j] = (step_mod * Σ_run[i, j] + Δ[i] * (θ[j] - μ_run[j])) / n
            end
        end
    end

    mean_accep = state.mean_accep + accepted / spl.average_interval

    new_state = AMGSStoppingState(
        state.ldf,
        state.transition,
        θ,
        state.logp,
        state.μ,
        state.kernel,
        step,
        state.β,
        state.stopped,
        mean_accep,
        μ_run,
        Σ_run,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )

    # End of a block: test the stopping criterion, then reset the accumulator.
    if step % spl.average_interval == 0
        stopped = check_end_adaptation(new_state, spl)
        new_state = AMGSStoppingState(
            new_state.ldf,
            new_state.transition,
            θ,
            new_state.logp,
            new_state.μ,
            new_state.kernel,
            step,
            new_state.β,
            stopped,
            0.0,                # reset the block acceptance average
            μ_run,
            Σ_run,
            new_state.Δ_cache,
            new_state.θ_prop,
            new_state.z_cache,
        )
    end

    return new_state
end

"Return `state` with only its step counter replaced."
function _with_step(state::AMGSStoppingState, step::Integer)
    return AMGSStoppingState(
        state.ldf,
        state.transition,
        state.θ,
        state.logp,
        state.μ,
        state.kernel,
        step,
        state.β,
        state.stopped,
        state.mean_accep,
        state.μ_running_avg,
        state.kernel_running_calc,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_Stopping_Sampler) = true
Turing.Inference.init_strategy(::AMGS_Stopping_Sampler) = DynamicPPL.InitFromPrior()

function _stopping_initial_state(
    rng::AbstractRNG, model::DynamicPPL.Model, initial_params
)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    return init_AMGSStoppingState(ldf, transition, θ, logp, initial_kernel(θ))
end

"""
    _stopping_step!(rng, state, spl, Σ, discard_sample)

One iteration: propose/accept, adapt the kernel (every iteration, exactly as
base AMGS does), then advance the block bookkeeping that drives the stopping
rule. Once adaptation has stopped, the kernel update is skipped and the chain
continues with a fixed proposal.
"""
function _stopping_step!(
    rng::AbstractRNG,
    state::AMGSStoppingState,
    spl::AMGS_Stopping_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)

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

    # Rebuild with the post-move position, log-density and transition.
    moved = AMGSStoppingState(
        state.ldf,
        transition,
        state.θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.β,
        state.stopped,
        state.mean_accep,
        state.μ_running_avg,
        state.kernel_running_calc,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )

    # Adapt every iteration while adaptation is live.
    if !moved.stopped
        _, β = update_kernel!(rng, moved, spl, prob)
        moved = AMGSStoppingState(
            moved.ldf,
            moved.transition,
            moved.θ,
            moved.logp,
            moved.μ,
            moved.kernel,
            moved.step,
            β,
            moved.stopped,
            moved.mean_accep,
            moved.μ_running_avg,
            moved.kernel_running_calc,
            moved.Δ_cache,
            moved.θ_prop,
            moved.z_cache,
        )
    end

    new_state = block_update_state(moved, accep, spl)

    return (discard_sample ? nothing : transition), new_state
end


function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _stopping_initial_state(rng, model, initial_params)
    return _stopping_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Sampler,
    state::AMGSStoppingState;
    discard_sample=false,
    kwargs...,
)
    return _stopping_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _stopping_initial_state(rng, model, initial_params)
    return _stopping_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Sampler,
    state::AMGSStoppingState;
    discard_sample=false,
    kwargs...,
)
    return _stopping_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::AMGS_Stopping_Sampler,
    state::AMGSStoppingState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSStoppingState(
        new_ldf,
        transition,
        θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.β,
        state.stopped,
        state.mean_accep,
        state.μ_running_avg,
        state.kernel_running_calc,
        state.Δ_cache,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::AMGS_Stopping_Sampler, state) = missing
