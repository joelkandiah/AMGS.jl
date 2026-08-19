"""
amgs_tempered_stopping_sample.jl

AMGS combining **tempering** with a **stopping rule** for the adaptation.

This is the union of the two variants:

- the target is the tempered posterior `temp·(log prior + log lik) + log|J|`
  (see [`TemperedLogJoint`](@ref)), and
- the Robbins–Monro adaptation freezes once the block-averaged acceptance rate
  settles inside `(lower_stopping_accep, upper_stopping_accep)`, or once
  `max_adaptation` iterations have elapsed (see
  [`AMGS_Stopping_Sampler`](@ref)).

Freezing the kernel matters more here than in the untempered case: a tempered
chain is often one rung of a tempering ladder, and swap moves between rungs are
only valid against a fixed, genuinely Markov kernel.

!!! warning
    With `temperature != 1` the stationary distribution is the *tempered*
    posterior, not the posterior.
"""

"""
    AMGS_Stopping_Tempered_Sampler(acceptance_rate, convergence_step;
                                   temperature=1.0,
                                   lower_stopping_accep=1.01,
                                   upper_stopping_accep=1.02,
                                   adaptation_interval=200,
                                   max_adaptation=-1)

Tempered AMGS whose adaptation stops once the acceptance rate settles.

# Fields
- `acceptance_rate`, `convergence_step`: adaptation target and Robbins–Monro
  decay exponent `γ`.
- `use_noise_stabiliser::Val{Bool}`, `noise_scaling`: optional covariance noise
  stabiliser (see [`add_noise!`](@ref)).
- `temperature`: tempering exponent applied to the posterior.
- `lower_stopping_accep`, `upper_stopping_accep`: acceptance-rate window that
  freezes the adaptation. The defaults `(1.01, 1.02)` lie above 1 and are
  therefore unreachable, which disables acceptance-based stopping unless the
  caller opts in.
- `adaptation_interval`: block length the acceptance rate is averaged over.
- `max_adaptation`: hard cap on adaptation iterations; `-1` disables the cap.

# Example
```julia
# temper, and freeze the kernel once acceptance settles in (0.15, 0.35)
spl = AMGS_Stopping_Tempered_Sampler(
    0.234, 0.6; temperature=0.5,
    lower_stopping_accep=0.15, upper_stopping_accep=0.35,
    adaptation_interval=500,
)
```
"""
struct AMGS_Stopping_Tempered_Sampler{
    T_accep<:Real,
    T_conv_step<:Real,
    T_scale<:Real,
    T_B,
    T_Temp<:Real,
    T_accep_l<:Real,
    T_accep_u<:Real,
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    temperature::T_Temp
    lower_stopping_accep::T_accep_l
    upper_stopping_accep::T_accep_u
    adaptation_interval::Int
    max_adaptation::Int
end

function AMGS_Stopping_Tempered_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool=false,
    noise_scaling::Real=1e-6;
    temperature::Real=1.0,
    lower_stopping_accep::Real=1.01,
    upper_stopping_accep::Real=1.02,
    adaptation_interval::Integer=200,
    max_adaptation::Integer=-1,
) where {T1<:Real,T2<:Real}
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    adaptation_interval > 0 ||
        throw(ArgumentError("adaptation_interval must be positive"))
    lower_stopping_accep < upper_stopping_accep ||
        throw(ArgumentError("lower_stopping_accep must be below upper_stopping_accep"))
    return AMGS_Stopping_Tempered_Sampler(
        acceptance_rate,
        convergence_step,
        Val(use_noise_stabiliser),
        noise_scaling,
        temperature,
        lower_stopping_accep,
        upper_stopping_accep,
        Int(adaptation_interval),
        Int(max_adaptation),
    )
end

"""
    AMGSStoppingTemperedState

State for [`AMGS_Stopping_Tempered_Sampler`](@ref): the tempered state plus the
block bookkeeping that drives the stopping rule.

`logp` caches the **tempered** log-density, which is what the acceptance ratio
uses. See [`AMGSStoppingState`](@ref) for the meaning of `stopped`,
`mean_accep`, `μ_running_avg` and `kernel_running_calc`.
"""
struct AMGSStoppingTemperedState{
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
    init_AMGSStoppingTemperedState(ldf, transition, θ, logp, kernel)

Build the initial tempered-stopping state, allocating every buffer once.
"""
function init_AMGSStoppingTemperedState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
)
    return AMGSStoppingTemperedState(
        ldf,
        transition,
        θ,
        logp,
        copy(θ),
        kernel,
        1,                  # step (never 0: η = (step+1)^-γ must stay below 1)
        zero(eltype(θ)),
        false,
        0.0,
        copy(θ),
        zero(kernel),
        similar(θ),
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel!(rng, state, spl, prob)

Robbins–Monro adaptation (Andrieu & Thoms Algorithm 4), identical to the other
AMGS variants — tempering changes the target, the stopping rule changes when
this runs, but the update itself is unchanged.
"""
function update_kernel!(
    rng::AbstractRNG,
    state::AMGSStoppingTemperedState,
    spl::AMGS_Stopping_Tempered_Sampler,
    prob::Real,
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

Whether to freeze the adaptation, given the acceptance rate averaged over the
block that has just completed. Called only at block boundaries.
"""
function check_end_adaptation(
    state::AMGSStoppingTemperedState, spl::AMGS_Stopping_Tempered_Sampler
)
    if (spl.max_adaptation != -1) && (state.step >= spl.max_adaptation)
        return true
    end
    if (state.mean_accep > spl.lower_stopping_accep) &&
        (state.mean_accep < spl.upper_stopping_accep)
        return true
    end
    return false
end

"Return `state` with only its step counter replaced."
function _with_step(state::AMGSStoppingTemperedState, step::Integer)
    return AMGSStoppingTemperedState(
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

"""
    block_update_state(state, accepted, spl)

Advance the block bookkeeping by one iteration: accumulate the block acceptance
rate and the within-block Welford mean/covariance, and at each block boundary
test the stopping criterion and reset the accumulator.

As in [`AMGS_Stopping_Sampler`](@ref), the accumulator is reset here rather than
in [`update_kernel!`](@ref); resetting it on every kernel update would cap it at
`1/adaptation_interval` and make the acceptance-rate criterion unreachable.
"""
function block_update_state(
    state::AMGSStoppingTemperedState,
    accepted::Bool,
    spl::AMGS_Stopping_Tempered_Sampler,
)
    if state.stopped
        return _with_step(state, state.step + 1)
    end

    step = state.step + 1
    θ = state.θ
    μ_run = state.μ_running_avg
    Σ_run = state.kernel_running_calc
    step_mod = (step - 1) % spl.adaptation_interval

    if step_mod == 0
        copyto!(μ_run, θ)
        fill!(Σ_run, zero(eltype(Σ_run)))
    else
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

    mean_accep = state.mean_accep + accepted / spl.adaptation_interval

    new_state = AMGSStoppingTemperedState(
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

    if step % spl.adaptation_interval == 0
        stopped = check_end_adaptation(new_state, spl)
        new_state = AMGSStoppingTemperedState(
            new_state.ldf,
            new_state.transition,
            θ,
            new_state.logp,
            new_state.μ,
            new_state.kernel,
            step,
            new_state.β,
            stopped,
            0.0,               # reset the block acceptance average
            μ_run,
            Σ_run,
            new_state.Δ_cache,
            new_state.θ_prop,
            new_state.z_cache,
        )
    end

    return new_state
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::AMGS_Stopping_Tempered_Sampler) = true
function Turing.Inference.init_strategy(::AMGS_Stopping_Tempered_Sampler)
    return DynamicPPL.InitFromPrior()
end

function _st_transition(state::AMGSStoppingTemperedState, accepted::Bool)
    return DynamicPPL.ParamsWithStats(
        state.θ, state.ldf, (; accepted=accepted, lp_tempered=state.logp)
    )
end

function _st_reuse(state::AMGSStoppingTemperedState, accepted::Bool)
    old = state.transition
    return DynamicPPL.ParamsWithStats(
        old.params, merge(old.stats, (; accepted=accepted, lp_tempered=state.logp))
    )
end

function _st_initial_state(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Tempered_Sampler,
    initial_params,
)
    ldf = build_ldf(model, rng, initial_params, TemperedLogJoint(spl.temperature))
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true, lp_tempered=logp))
    return init_AMGSStoppingTemperedState(ldf, transition, θ, logp, initial_kernel(θ))
end

"""
    _st_step!(rng, state, spl, Σ, discard_sample)

One iteration: propose and accept/reject against the tempered target, adapt the
kernel (every iteration, while adaptation is live), then advance the block
bookkeeping that drives the stopping rule.
"""
function _st_step!(
    rng::AbstractRNG,
    state::AMGSStoppingTemperedState,
    spl::AMGS_Stopping_Tempered_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)

    moved = AMGSStoppingTemperedState(
        state.ldf,
        state.transition,
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

    # The transition cache is kept consistent with the position regardless of
    # `discard_sample`, which only suppresses what is returned.
    transition = accep ? _st_transition(moved, true) : _st_reuse(moved, false)

    β = moved.β
    if !moved.stopped
        _, β = update_kernel!(rng, moved, spl, prob)
    end

    moved = AMGSStoppingTemperedState(
        moved.ldf,
        transition,
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

    new_state = block_update_state(moved, accep, spl)
    return (discard_sample ? nothing : transition), new_state
end


function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Tempered_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _st_initial_state(rng, model, spl, initial_params)
    return _st_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Tempered_Sampler,
    state::AMGSStoppingTemperedState;
    discard_sample=false,
    kwargs...,
)
    return _st_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Tempered_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _st_initial_state(rng, model, spl, initial_params)
    return _st_step!(rng, state, spl, state.kernel, 1.0, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Stopping_Tempered_Sampler,
    state::AMGSStoppingTemperedState;
    discard_sample=false,
    kwargs...,
)
    return _st_step!(rng, state, spl, state.kernel, exp(state.β), discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::AMGS_Stopping_Tempered_Sampler,
    state::AMGSStoppingTemperedState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return AMGSStoppingTemperedState(
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

function getlogevidence(transitions, sampler::AMGS_Stopping_Tempered_Sampler, state)
    return missing
end
