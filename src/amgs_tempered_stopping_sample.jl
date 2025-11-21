#  write a functions step and step_warmup which are used in the AMGS_Stopping_Tempered_sampler.
# Based on Algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y

# Start by defining the sampler

"""
    AMGS_Stopping_Tempered_Sampler(acceptance_rate, convergence_step; use_noise_stabiliser=false, noise_scaling=1e-6, temperature=1, lower_stopping_accep=1.01, upper_stopping_accep=1.02, adaptation_interval=200, max_adaptation=-1)

Tempered version of the Adaptive Metropolis-within-Gibbs sampler with automatic stopping criteria.

This sampler combines tempered MCMC with adaptive stopping based on acceptance rate
convergence. Adaptation stops when the running acceptance rate stays within the bounds
[`lower_stopping_accep`, `upper_stopping_accep`] for `adaptation_interval` consecutive steps,
or after `max_adaptation` steps if specified.

# Fields
- `acceptance_rate::Real`: target acceptance rate for the Robbins–Monro update.
- `convergence_step::Real`: controls the step-size decay in adaptation (power).
- `use_noise_stabiliser::Val{Bool}`: fast-dispatch flag to optionally add noise.
- `noise_scaling::Real`: variance scale used by the optional noise stabiliser.
- `temperature::Real`: temperature parameter for likelihood scaling.
- `lower_stopping_accep::Real`: lower bound for stopping acceptance rate check.
- `upper_stopping_accep::Real`: upper bound for stopping acceptance rate check.
- `adaptation_interval::Int`: number of steps to check for acceptance rate stability.
- `max_adaptation::Int`: maximum adaptation steps (-1 for unlimited).

# Examples
```julia
sampler = AMGS_Stopping_Tempered_Sampler(0.234, 1000, temperature=0.8)
chain = sample(model, sampler, 10000)  # May stop early when converged
```

TODO: Add detailed explanation of stopping criteria and convergence detection.
TODO: Explain the interaction between temperature and stopping conditions.
"""
struct AMGS_Stopping_Tempered_Sampler{ T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_B, T_Temp <: Real} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    temperature::T_Temp
    lower_stopping_accep::T_accep
    upper_stopping_accep::T_accep
    adaptation_interval::Int
    max_adaptation::Int
end

function AMGS_Stopping_Tempered_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
) where {T1 <: Real, T2 <: Real}
return AMGS_Stopping_Tempered_Sampler(acceptance_rate, convergence_step, Val(false), 1e-6, 1, 1.01,1.02, 200, -1)
end

function AMGS_Stopping_Tempered_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real }
return AMGS_Stopping_Tempered_Sampler(acceptance_rate, convergence_step, Val(use_noise_stabiliser), noise_scaling, 1, 1.01, 1.02, 200, -1)
end


DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_Stopping_Tempered_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
"""
    AMGSStoppingTemperedState

State container used by the AMGS Stopping Tempered sampler to store running statistics
required for adaptation and automatic stopping in tempered MCMC.

# Fields
- `transition`: the latest `Turing.Inference.Transition` object.
- `varinfo`: the `VarInfo` corresponding to the current parameter values.
- `μ`: running mean vector of parameters.
- `kernel`: running covariance (or scale) matrix used for proposals.
- `step`: integer step counter for adaptation scheduling.
- `β`: scalar adapt scaling for acceptance rate control.
- `logjoint_internal`: cached log joint density for the current state.
- `logjoint_tempered`: cached tempered log joint density (includes temperature scaling).
- `mean_accep`: running average of acceptance rates for stopping criteria.
- `μ_running_avg`: running average of parameter means for convergence monitoring.
"""
struct AMGSStoppingTemperedState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    Tμ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tβ<:Real,
    TL<:Real,
    TTemp,
    # Add mean acceptance
    TAccep<:Real,
}
    transition::TTrans
    varinfo::TVI
    μ::Tμ
    kernel::TKernel
    step::TStep
    β::Tβ
    logjoint_internal::TL
    logjoint_tempered::TTemp
    mean_accep::TAccep
    μ_running_avg::Tμ
    kernel_running_calc::TKernel
    stopped::Bool
end

# Define the initial state of the sampler
"""
    init_AMGSStoppingTemperedState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS Tempered sampler with the given transition, mean vector μ, and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
- `temp::T_Temp`: The temperature for tempering.
"""
function init_AMGSStoppingTemperedState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel,
    temp::T_Temp,
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix, T_Temp <: Real}
    ## copy vector length of mu and initialise zero vector for beta
    step = 0
    β = zero(eltype(μ))
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    t_lp = temp * (DynamicPPL.getlogprior(varinfo) + DynamicPPL.getloglikelihood(varinfo)) - DynamicPPL.getlogjac(varinfo)
    AMGSStoppingTemperedState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(β), typeof(lp), typeof(t_lp), Float64}(transition, varinfo, μ, kernel, step, β, lp, t_lp, 0.0, μ, zero(kernel), false)
end

"""
    init_AMGSStoppingTemperedState(transition::T_trans, μ::T_μ, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: AbstractMatrix}

Initialize the state of the AMGS Tempered sampler with the given transition, mean value μ, and kernel matrix for a 1D case.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean value.
- `kernel::T_kernel`: The initial kernel matrix.
- `temp::T_Temp`: The temperature for tempering.
"""
function init_AMGSStoppingTemperedState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel,
    temp::T_Temp
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix, T_Temp <: Real}
    ## copy vector length of mu and initialise zero vector for beta
    step = 0
    β = zero(typeof(μ))
    new_μ = [μ]
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    t_lp = temp * (DynamicPPL.getlogprior(varinfo) + DynamicPPL.getloglikelihood(varinfo)) - DynamicPPL.getlogjac(varinfo)
    AMGSStoppingTemperedState{T_trans, TVI, typeof(new_μ), T_kernel, Int, typeof(β), typeof(lp), typeof(t_lp), Float64}(transition, varinfo, new_μ, kernel, step, β, lp, t_lp, 0.0, new_μ, zero(kernel), false)
end

"""
    update_kernel(rng::TRNG, state::AMGSStoppingTemperedState, trans::T, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real, TRNG <: AbstractRNG}

Update the kernel matrix, and internal quantities of the AMGS Tempered sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSStoppingTemperedState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(rng::TRNG, state::AMGSStoppingTemperedState, trans::T, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real, TRNG <: AbstractRNG}
    μ    = state.μ_running_avg
    step = state.step
    Δ    = state.varinfo[:]
    n    = length(Δ)

    # Robbins–Monro gain
    @fastmath begin
        η = (step + 1.0)^(-spl.alg.convergence_step)

        # Adapt acceptance-rate scaling
        β = state.β + η * (prob - spl.alg.acceptance_rate)

        # Update running mean μ ← (1−η)μ + ηθ
        @. μ += η * (Δ - μ)

        # Δ = θ - μ (store in preallocated buffer Δ in the state)
        @. Δ -= μ

        # Outer product update, symmetric and in-place:
        # Optimized: Use BLAS symmetric rank-1 update for outer product
        # First scale kernel by (1-η)
        state.kernel .*= (1.0 - η)
        # Then add η*Δ*Δᵀ using symmetric rank-1 update
        BLAS.syr!('U', η, Δ, state.kernel)
        # Copy upper triangle to lower triangle for symmetry
        @inbounds for j in 1:n
            for i in (j+1):n
                state.kernel[i,j] = state.kernel[j,i]
            end
        end

        # Optional noise stabilisation (fast Val dispatch)
        @inline _add_noise!(rng, state.kernel, spl.alg.noise_scaling, n, spl.alg.use_noise_stabiliser)
    end 

    ## Reset the running covariance and mean calculation
    return state.kernel, AMGSStoppingTemperedState(trans, state.varinfo, μ, state.kernel, step, β, state.logjoint_internal, state.logjoint_tempered, 0.0, zero(μ), zero(state.kernel), state.stopped)
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::AMGS_Stopping_Tempered_Sampler) = true 

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:AMGSStoppingTemperedState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:AMGS_Stopping_Tempered_Sampler},
    state::AMGSStoppingTemperedState,
    params::AbstractVarInfo,
)

    # Update the vi
    # vi = state.varinfo
    # vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, params)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    tempered_lp = sampler.alg.temperature * (DynamicPPL.getlogprior(vi) + DynamicPPL.getloglikelihood(vi)) - DynamicPPL.getlogjac(vi)
    # update the model state
    new_transition = Turing.Inference.Transition(model, vi, (; lp_tempered = tempered_lp, lp_internal = lp))
    new_state = AMGSStoppingTemperedState(new_transition, vi, state.μ, state.kernel, state.step, state.β, lp, tempered_lp, state.mean_accep, state.μ_running_avg, state.kernel_running_calc, state.stopped) 
    
    # Update the VarInfo with the new log density
    return new_state
end

# Function to check whether to stop adapting
"""
    check_end_adaptation(state, accepted, spl)

Check whether adaptation should stop based on the current state and acceptance criteria.

# Arguments
- `state::AMGSStoppingTemperedState`: The current sampler state.
- `accepted::Bool`: Whether the current proposal was accepted.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS Stopping Tempered sampler.

# Returns
`true` if adaptation should stop, `false` otherwise.
"""
function check_end_adaptation(state::AMGSStoppingTemperedState, accepted::Bool, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler})
    # Only check if this step we are at the end of an adaptation interval
    if ((state.step + 1) % spl.alg.adaptation_interval) !== 0
        return false
    end
    
    # End if we have hit the max adaptation steps
    # -1 indicates no max
    if (spl.alg.max_adaptation != -1) && (state.step >= spl.alg.max_adaptation)
        return true
    end

    # Check whether to stop adapting
    if (state.mean_accep > spl.alg.lower_stopping_accep) && (state.mean_accep < spl.alg.upper_stopping_accep)
        return true
    end
    return false
end

# update the runing mean and covariance
# Use Welford's online algortihm for numerical stability
"""
    running_update_state(state, step_mod, accepted, spl)

Update the running statistics for the tempered stopping sampler using Welford's online algorithm.

# Arguments
- `state::AMGSStoppingTemperedState`: The current sampler state.
- `step_mod::Int`: The step modulo for tracking adaptation intervals.
- `accepted::Bool`: Whether the current proposal was accepted.
- `spl`: The sampler.

# Returns
The updated state with new running statistics.
"""
function running_update_state(state::AMGSStoppingTemperedState, step_mod::Int, accepted::Bool, spl)
    if state.stopped
        return state
    end
    # Check whether to stop adapting
    stop_adapting = check_end_adaptation(state, accepted, spl)
    theta = state.varinfo[:]
    if step_mod == 0
        state = AMGSStoppingTemperedState(
            state.transition,
            state.varinfo,
            state.μ,
            state.kernel,
            state.step + 1,
            state.β,
            state.logjoint_internal,
            state.logjoint_tempered,
            state.mean_accep + accepted /spl.alg.adaptation_interval,
            theta,
            state.kernel,
            stop_adapting
        )
    else     
        n = step_mod + 1
        δ = theta .- state.μ_running_avg
        running_μ = state.μ_running_avg .+ δ ./ n
        state = AMGSStoppingTemperedState(
            state.transition,
            state.varinfo,
            state.μ,
            state.kernel,
            state.step + 1,
            state.β,
            state.logjoint_internal,
            state.logjoint_tempered,
            state.mean_accep + accepted /spl.alg.adaptation_interval,
            running_μ,
            (step_mod * state.kernel_running_calc .+ δ * transpose(theta .- running_μ)) ./ n,
            stop_adapting
        )
    end
    return state
end


# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGS Tempered sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `state::AMGSStoppingTemperedState`: The current state of the sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Tempered_Sampler},
    state::AMGSStoppingTemperedState,
    Σ::AbstractMatrix
)
    # Grab the parameters we are performing inference on
    θ = vi[:]
    # Propose a new set of parameters
    proposal = proposewalk(rng, θ, Σ)
    
    proposal_vi = DynamicPPL.unflatten(vi, proposal)

    # Call a function to update the varinfo and evaluate the logdensity
    return Turing.Inference.setparams_varinfo!!(model, spl, state, proposal_vi) 
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGS Tempered sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Tempered_Sampler};
    initial_params=nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = DynamicPPL.default_varinfo(rng, model, spl)

    # If there are initial parameters, set them and the logdensity
    if initial_params !== nothing
        vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
        vis = last(DynamicPPL.evaluate!!(model, vis))
    end

    # Link the variables
    DynamicPPL.link!!(vis, model)
    vis = last(DynamicPPL.evaluate!!(model, vis))

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)
    prev_transition = Turing.Inference.Transition(model, vis, (; lp_tempered = spl.alg.temperature * (DynamicPPL.getlogprior(vis) + DynamicPPL.getloglikelihood(vis)) - DynamicPPL.getlogjac(vis), lp_internal = lp_old))

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    # get the old state:
    orig_state = init_AMGSStoppingTemperedState(prev_transition, vis, theta, Σ, spl.alg.temperature)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(
        rng,
        new_state.logjoint_tempered,
        orig_state.logjoint_tempered,
    )
    # accep = mh_accept_logdens_temp(rng, lp_proposal, lp_old)

    return accep ? new_state.transition : prev_transition, accep ? new_state : orig_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGS Tempered sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `state::AMGSStoppingTemperedState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Tempered_Sampler},
    state::AMGSStoppingTemperedState;
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.varinfo
    # lp_old = state.logjoint_internal

    # Get the kernel
    Σ = state.kernel

    # Propose the new parameters and get the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, Σ)
    # lp_proposal = proposal_state.logjoint_internal 

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, proposal_state.logjoint_tempered, state.logjoint_tempered)

    return accep ? proposal_state.transition : state.transition, accep ? proposal_state : state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGS Tempered sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_Stopping_Tempered_Sampler};
    initial_params=nothing,
    kwargs...,
)
    vis = DynamicPPL.default_varinfo(rng, model, spl)

    # If there are initial parameters, set them and the logdensity
    if initial_params !== nothing
        vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
        vis = last(DynamicPPL.evaluate!!(model, vis))
    end    # Link the variables

    vis = DynamicPPL.link!!(vis, model)
    vis = last(DynamicPPL.evaluate!!(model, vis))

    # Get the log density of the current parameters
    # lp_old = DynamicPPL.getlogjoint_internal(vis)

    # Create a transition object in case of rejection
    prev_transition = Turing.Inference.Transition(model, vis, (;lp_tempered = spl.alg.temperature * (DynamicPPL.getlogprior(vis) + DynamicPPL.getloglikelihood(vis)) - DynamicPPL.getlogjac(vis), lp_internal = DynamicPPL.getlogjoint_internal(vis)))

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    orig_state = init_AMGSStoppingTemperedState(prev_transition, vis, theta, Σ, spl.alg.temperature)
    lp_old = orig_state.logjoint_tempered
    # Propose a new set of parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = proposal_state.logjoint_tempered

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old) 

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : prev_transition

    # Construct the initial state of the sampler
    # new_state = init_AMGSStoppingTemperedState(new_trans, theta, Σ)
    upd_state = accep ? proposal_state : orig_state
    
    # Update the running mean and covariance and check whether the adaptation should stop
    upd_state = running_update_state(upd_state, upd_state.step % spl.alg.adaptation_interval, accep, spl)

    # If stopped, do not update the kernel
    if upd_state.stopped
        return new_trans, upd_state
    end

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    
    if (upd_state.step % spl.alg.adaptation_interval) == 0
        _, upd_state = update_kernel(rng, upd_state, new_trans, spl, prob)
    end

    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGS Tempered sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `state::AMGSStoppingTemperedState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Tempered_Sampler},
    state::AMGSStoppingTemperedState;
    kwargs...
)
    vis = state.varinfo
    lp_old = state.logjoint_tempered
    # Propose the new parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, exp(state.β) .* state.kernel)
    lp_proposal = proposal_state.logjoint_tempered

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : state.transition 
    new_state = accep ? proposal_state : state

    # Update the running mean and covariance and check whether the adaptation should stop
    new_state = running_update_state(new_state, new_state.step % spl.alg.adaptation_interval, accep, spl)

    #if stopped, do not update the kernel
    if !new_state.stopped
        return new_trans, new_state
    end
    
    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    _, new_state = update_kernel(rng, new_state, new_trans, spl, prob)
    return new_trans, new_state
end


# Define assume functions for the AMGS Tempered sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler}, dist::Distribution, vn::DynamicPPL.VarName, vi
)
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end

# Define function to get log evidence
"""
    getlogevidence(transitions, sampler, state)

Get the log evidence from a collection of transitions. For AMGS samplers, this returns `missing` as log evidence computation is not implemented.

# Arguments
- `transitions`: Vector of transitions from the sampling process.
- `sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS Stopping Tempered sampler.
- `state`: The final sampler state.

# Returns
`missing` (log evidence computation not implemented for AMGS samplers).
"""
getlogevidence(transitions, sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:Turing.Inference.Transition}, model::AbstractMCMC.AbstractModel, spl::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGS Tempered sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:Turing.Inference.Transition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler}`: The AMGS sampler.
- `state`: The current state of the sampler.
- `chain_type::Type{MCMCChains.Chains}`: The type of chain.
- `save_state`: Whether to save the state (default: false).
- `stats`: The statistics (default: missing).
- `sort_chain`: Whether to sort the chain (default: false).
- `include_varname_to_symbol`: Whether to include variable name to symbol mapping (default: true).
- `discard_initial`: The number of initial samples to discard (default: 0).
- `thinning`: The thinning interval (default: 1).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.bundle_samples(
    ts::Vector{<:Turing.Inference.Transition},
    model::AbstractMCMC.AbstractModel,
    spl::DynamicPPL.Sampler{<:AMGS_Stopping_Tempered_Sampler},
    state,
    chain_type::Type{MCMCChains.Chains};
    save_state=false,
    stats=missing,
    sort_chain=false,
    include_varname_to_symbol=true,
    discard_initial=0,
    thinning=1,
    kwargs...,
)

    # Convert transitions to array format.
    # Also retrieve the variable names.
    varnames, vals = _params_to_array(model, ts)
    varnames_symbol = map(Symbol, varnames)

    # Get the values of the extra parameters in each transition.
    extra_params, extra_values = Turing.Inference.get_transition_extras(ts)

    # Extract names & construct param array.
    nms = [varnames_symbol; extra_params]
    parray = hcat(vals, extra_values)

    # Get the average or final log evidence, if it exists.
    le = getlogevidence(ts, spl, state)

    # Set up the info tuple.
    info = NamedTuple()

    if include_varname_to_symbol
        info = merge(info, (varname_to_symbol=OrderedDict(zip(varnames, varnames_symbol)),))
    end

    if save_state
        info = merge(info, (model=model, sampler=spl, samplerstate=state))
    end

    # Merge in the timing info, if available
    if !ismissing(stats)
        info = merge(info, (start_time=stats.start, stop_time=stats.stop))
    end

    # Conretize the array before giving it to MCMCChains.
    parray = MCMCChains.concretize(parray)

    # Chain construction.
    chain = MCMCChains.Chains(
        parray,
        nms,
        (internals=extra_params,);
        evidence=le,
        info=info,
        start=discard_initial + 1,
        thin=thinning,
    )

    return sort_chain ? sort(chain) : chain
end

