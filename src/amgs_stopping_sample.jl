# write a functions step and step_warmup which are used in the AMGS_Stopping_sampler.
# Write code based on algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y combined with the extensions in the appendix of https://arxiv.org/abs/2408.04178 (note appendix to be published soon)

# Start by defining the sampler
# Note added the stopping conditions as parameters to the sampler
# The max number of steps where adaptation occurs, the lower and upper bounds of the acceptance rate to check at each average interval
struct AMGS_Stopping_Sampler{T_accep <: Real, T_conv_step <: Real, T_adapt_period <: Integer, T_accep_l <: Real, T_accep_u <: Real, T_accep_interval <: Integer, T_scale <: Real} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    max_adapt_period::T_adapt_period
    lower_stopping_accep::T_accep_l
    upper_stopping_accep::T_accep_u
    average_interval::T_accep_interval
    use_noise_stabiliser::Bool
    noise_scaling::T_scale
end

"""
    AMGS_Stopping_Sampler(acceptance_rate, convergence_step, max_adapt_period, lower_stopping_accep, upper_stopping_accep, average_interval, space::Tuple; noise_stabiliser = false, noise_stabiliser_scaling = 1e-6)

Construct an `AMGS_Stopping_Sampler` object.

# Arguments
- `acceptance_rate::Real`: The target acceptance rate.
- `convergence_step::Real`: A tuning parameter for convergence/step size.
- `max_adapt_period::Integer`: The maximum adaptation period.
- `lower_stopping_accep::Real`: The lower bound for stopping acceptance rate.
- `upper_stopping_accep::Real`: The upper bound for stopping acceptance rate.
- `average_interval::Integer`: The interval for averaging acceptance rates.
- `noise_stabiliser::Bool`: (Optional) Whether to use a noise stabiliser, defaults to false.
- `noise_stabiliser_scaling::Real`: (Optional) The scaling of the noise stabiliser, defaults to 1e-6.

# Returns
An `AMGS_Stopping_Sampler` instance.
"""
function AMGS_Stopping_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    max_adapt_period::T3,
    lower_stopping_accep::T4,
    upper_stopping_accep::T5,
    average_interval::T6;
) where {T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer}
    return AMGS_Stopping_Sampler{T1, T2, T3, T4, T5, T6, Float64}(acceptance_rate, convergence_step, max_adapt_period, lower_stopping_accep, upper_stopping_accep, average_interval, false, 1e-6)
end

DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_Stopping_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
# In addition to the AMGSState, we also store the previous accepts as well as whether to adaptation has ended
"""
    AMGSStoppingState{TTrans<:Turing.Inference.Transition, Tμ<:Vector, TKernel<:Matrix, TStep<:Integer, Tβ<:Real, TΔi<:Vector}

The state object for the AMGS Stopping sampler.

# Fields
- `transition::TTrans`: The current transition object.
- `μ::Tμ`: The current mean vector of the target distribution.
- `kernel::TKernel`: The current covariance kernel.
- `step::TStep`: The current step number.
- `β::Tβ`: The global scale parameter.
- `Δi::TΔi`: The vector of recent acceptance values.
- `stopped::Bool`: Whether the adaptation process has stopped.
"""
struct AMGSStoppingState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    Tμ<:Vector,
    TKernel<:Matrix,
    TStep<:Integer,
    Tβ<:Real,
    TΔi<:Vector,
    TL<:Real
}
    transition::TTrans
    varinfo::TVI
    μ::Tμ
    kernel::TKernel
    step::TStep
    β::Tβ
    Δi::TΔi
    stopped::Bool
    logjoint_internal::TL
end

# Define the initial state of the sampler
"""
    init_AMGSStoppingState(transition, μ, kernel, sampler)

Initialize the `AMGSStoppingState` object.

# Arguments
- `transition::Turing.Inference.Transition`: The initial transition object.
- `μ::Vector{<:Real}`: The initial mean vector.
- `kernel::Matrix{<:Real}`: The initial kernel matrix.
- `sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.

# Returns
An `AMGSStoppingState` instance.
"""
function init_AMGSStoppingState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel,
    sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    Δi = zeros(eltype(μ), sampler.alg.average_interval)
    lp = DynamicPPL.getlogjoint_internal(varinfo)
    AMGSStoppingState{T_trans, TVI, Vector{T_μ}, T_kernel, Int, typeof(β), typeof(Δi), typeof(lp)}(transition, varinfo, μ, kernel, step, β, Δi, false, lp)
end

# Define the initial state of the sampler in a 1D case
"""
    init_AMGSStoppingState(transition, μ, kernel, sampler)

Initialize the `AMGSStoppingState` object for a 1D case.

# Arguments
- `transition::Turing.Inference.Transition`: The initial transition object.
- `μ::Real`: The initial mean value.
- `kernel::Matrix{<:Real}`: The initial kernel matrix.
- `sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.

# Returns
An `AMGSStoppingState` instance.
"""
function init_AMGSStoppingState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel,
    sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    μ = [μ]
    Δi = zeros(typeof(μ), sampler.alg.average_interval)
    lp = DynamicPPL.getlogjoint_internal(varinfo)
    AMGSStoppingState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(β),typeof(Δi), typeof(lp)}(transition, varinfo, μ, kernel, step, β, Δi, false, lp)
end

# Write a function which updates the kernel
"""
    update_kernel(state, trans, spl, prob)

Update the kernel matrix, mean vector, and other internal quantities of the `AMGSStoppingState`.

# Arguments
- `state::AMGSStoppingState`: The current state of the sampler.
- `trans::Turing.Inference.Transition`: The new transition object.
- `spl::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `prob::Real`: The acceptance probability.

# Returns
A new `AMGSStoppingState` with updated quantities.
"""
function update_kernel(state::AMGSStoppingState, trans::T, spl::Sampler{<:AMGS_Stopping_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real}
    # Grab useful quantities from the state
    Σ = state.kernel
    μ = state.μ
    step = state.step
    β = state.β

    # Follow maths from arXiv paper to update the kernel and the global scale β
    η = (step + 1.0)^(-spl.alg.convergence_step)
    β = β + η * (prob - spl.alg.acceptance_rate)
    Δ = state.varinfo[:] - μ
    μ = (1.0 - η) * μ + (η * state.varinfo[:])
    
    # Add noise to the kernel if requested
    noise_term = spl.alg.use_noise_stabiliser * spl.alg.noise_scaling * η * rand(size(Δ)[1])
    Σ = (1.0 - η) * Σ + (η * (Δ * (Δ'))) + diagm(noise_term)
    new_state = AMGSStoppingState(trans, state.varinfo, μ, Σ, step + 1, β, state.Δi, state.stopped, state.logjoint_internal)
    return new_state
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::AMGS_Stopping_Sampler) = true 

# Get varinfo
function get_varinfo(state::T) where {T<:AMGSStoppingState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:AMGS_Stopping_Sampler},
    state::AMGSStoppingState,
    params::TVI,
    ) where {TVI <: AbstractVarInfo}

    # Update the vi
    vi = state.varinfo
    vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, vi)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = Turing.Inference.Transition(model, vi, nothing)
    new_state = AMGSStoppingState(new_transition, vi, state.μ, state.kernel, state.step, state.β, state.Δi, state.stopped, lp) 
    
    # Update the VarInfo with the new log density
    return new_state
end


# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng, vi, model, spl, Σ)

Propose a new set of parameters using a random walk proposal with the given covariance matrix.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information object.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `state::AMGSStoppingState`: The current state of the sampler.
- `Σ::AbstractMatrix{<:Real}`: The covariance matrix for the proposal.

# Returns
The updated `VarInfo` object with the proposed parameters and log density.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler},
    state::AMGSStoppingState,
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

# Write a function which initialises the chain where there are warmup steps
"""
    step_warmup(rng, model, spl; initial_params = nothing, kwargs...)

Perform an initial warmup step for the AMGS Stopping sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `initial_params`: (Optional) The initial parameters.
- `kwargs...`: Additional keyword arguments.

# Returns
A tuple containing the transition and the updated state.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler};
    initial_params = nothing,
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
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = 0.01 .* diagm(abs.(theta)) + (1e-6 * I)

    # get the old state:
    orig_state = init_AMGSStoppingState(prev_transition, vis, theta, Σ, spl)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, orig_state 
    end

    return new_state.transition, new_state 

end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Sampler}, state::AMGSStoppingState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGS Stopping sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `state::AMGSStoppingState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler},
    state::AMGSStoppingState;
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Get the kernel
    Σ = state.kernel

    # Propose the new parameters and get the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, Σ)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return state.transition, state
    end

    return proposal_state.transition, proposal_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_Stopping_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGS Stopping sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_Stopping_Sampler};
    initial_params = nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.

    vis = DynamicPPL.default_varinfo(rng, model, spl)

    # If there are initial parameters, set them and the logdensity
    if initial_params !== nothing
        vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
        vis = last(DynamicPPL.evaluate!!(model, vis))
    end    # Link the variables

    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)

    # Create a transition object in case of rejection
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = 0.01 .* diagm(abs.(theta)) + (1e-6 * I)

    orig_state = init_AMGSStoppingState(prev_transition, vis, theta, Σ, spl) 
    # Propose a new set of parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    if !accep
        new_trans = prev_transition
    else
        new_trans = proposal_state.transition
    end

    # Update Δi if accepted and there is a non-zero averaging interval
    if accep && spl.alg.average_interval >= 1
        # (Want to update only the Δi[1] element as we can guarantee adaptation has started as long as the averaging interval is greater than 0)
        new_state_Δi = copy(proposal_state.Δi)
        new_state_Δi[1] = 1
        new_state = AMGSStoppingState(proposal_state.transition, proposal_state.varinfo, proposal_state.μ, proposal_state.kernel, proposal_state.step, proposal_state.β, new_state_Δi, proposal_state.stopped, proposal_state.logjoint_internal)
    end

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    upd_state = update_kernel(new_state, new_trans, spl, prob)

    return new_trans, upd_state
end


#TODO Add a case for when the averaging interval is Inf

"""
    AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Sampler}, state::AMGSStoppingState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGS Stopping sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `state::AMGSStoppingState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler},
    state::AMGSStoppingState;
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Propose the new parameters and get the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, exp(state.β) * state.kernel)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    new_trans = accep ? proposal_state.transition : state.transition
        
    # Update Δi with the new acceptance
    upd_Δi = vcat(state.Δi[2:end], [accep])

    # Calculate the acceptance probability
    prob = exp(min(0.0, (lp_proposal - lp_old)))

    # Determine if the adaptation should stop
    stopped = state.stopped || (spl.alg.average_interval == 0) ||
        (state.step + 1 >= spl.alg.max_adapt_period) ||
        ifelse(ismissing(spl.alg.average_interval), false, ((sum(upd_Δi) / spl.alg.average_interval <= spl.alg.upper_stopping_accep) &&
            (sum(upd_Δi) / spl.alg.average_interval >= spl.alg.lower_stopping_accep) &&
            (((state.step + 1) % spl.alg.average_interval) == 0)))

    # Create a new state object
    new_state = AMGSStoppingState(new_trans, accep ? proposal_state.varinfo : state.varinfo, state.μ, state.kernel, state.step, state.β, upd_Δi, stopped, accep ? proposal_state.logjoint_internal : state.logjoint_internal)

    # Update the kernel if adaptation has not stopped
    if !new_state.stopped
        new_state = update_kernel(new_state, new_trans, spl, prob)
    end

    return new_trans, new_state
end

function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}, dist::Distribution, vn::T_VN, vi
) where {T_VN <: DynamicPPL.VarName}
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end

getlogevidence(transitions, sampler::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler}, state) = missing

function AbstractMCMC.bundle_samples(
    ts::Vector{<:Turing.Inference.Transition},
    model::AbstractMCMC.AbstractModel,
    spl::DynamicPPL.Sampler{<:AMGS_Stopping_Sampler},
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
