# write functions step and step_warmup which are used in the RAM_sampler.
# Using the algorithm constructed in https://arxiv.org/abs/1011.4381

# Start by defining the sampler

struct RAM_Sampler{T_accep <: Real, T_conv_step <: Real, T_lb_eig <: Real, T_ub_eig <: Real} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    lb_eig::T_lb_eig
    ub_eig::T_ub_eig
    bound_eigs::Bool
end

"""
    RAM_Sampler(acceptance_rate::T1, convergence_step::T2, space::Tuple; lb_eig::T3 = -Inf, ub_eig::T4 = Inf, bound_eigs::Bool = false) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}

Create a new RAM sampler with the specified acceptance rate, convergence step, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `convergence_step::T2`: A tuning parameter for the rate of convergence/step size of the sampler.
- `lb_eig::T3`: The lower bound for the eigenvalues for the acceptance of the kernel (default: -Inf).
- `ub_eig::T4`: The upper bound for the eigenvalues for the acceptance of the kernel (default: Inf).
- `bound_eigs::Bool`: Whether to bound the eigenvalues (default: false).
"""
function RAM_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
) where {T1 <: Real, T2 <: Real}
    return RAM_Sampler{T1, T2, Float64, Float64}(acceptance_rate, convergence_step, -Inf, Inf, false)
end

# DynamicPPL.default_chain_type(sampler::RAM_Sampler) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
"""
    RAMState

State container used by the RAM (Robust Adaptive Metropolis) sampler to store running statistics
required for adaptation using Cholesky-based proposals.

# Fields
- `transition`: the latest `Turing.Inference.Transition` object.
- `varinfo`: the `VarInfo` corresponding to the current parameter values.
- `kernel`: Cholesky factor of the running covariance matrix used for proposals.
- `step`: integer step counter for adaptation scheduling.
- `logjoint_internal`: cached log joint density for the current state.
"""
struct RAMState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    TKernel<:Cholesky,
    TStep<:Integer,
    TL<:Real,
}
    transition::TTrans
    varinfo::TVI
    kernel::TKernel
    step::TStep
    logjoint_internal::TL
end

# Define the initial state of the sampler
"""
    init_RAMState(transition::T_trans, kernel::T_kernel) where {T_trans <: AdvancedMH.AbstractTransition, T_kernel <: Cholesky}

Initialize the state of the RAM sampler with the given transition and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_RAMState(
    transition::T_trans,
    varinfo::TVI,
    kernel::T_kernel,
) where {T_trans <: Turing.Inference.Transition, TVI<:AbstractVarInfo, T_kernel <: Cholesky}
    # Initialize step
    step = 1
    # Initialize log joint density
    DynamicPPL.getlogjoint_internal(varinfo)
    RAMState{T_trans, TVI, T_kernel, Int, typeof(lp)}(transition, varinfo, kernel, step, lp)
end

# Write a function which updates the kernel
"""
    update_kernel(state::RAMState, trans::Turing.Inference.Transition, U::T, spl::RAM_Sampler, prob::T3) where {T <: Vector, T3 <: Real}

Update the kernel matrix of the RAM sampler state based on the new transition and acceptance probability.

# Arguments
- `state::RAMState`: The current state of the sampler.
- `trans::Turing.Inference.Transition`: The new transition object.
- `U::T`: The proposed update vector.
- `spl::RAM_Sampler`: The RAM sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::RAMState, trans::Turing.Inference.Transition, U::T, spl::RAM_Sampler, prob::T3) where {T <: Vector, T3 <: Real}
    # Grab useful quantities from the state
    S = state.kernel
    step = state.step

    # Follow maths from the paper to update the kernel
    # Do this via the Cholesky update and downdate to simplify calculations
    η = (step + 1.0)^(-spl.convergence_step)
    accepdiff = (prob - spl.acceptance_rate)
    β = sqrt(η * abs(accepdiff)) * S.L * U / norm(U)

    S = if sign(accepdiff) == 1
        LinearAlgebra.lowrankupdate(S, β)
    else
        LinearAlgebra.lowrankdowndate(S, β)
    end

    # Bound the eigenvalues if required
    if spl.bound_eigs
        S = ifelse(
            all(x -> spl.lb_eig <= x <= spl.ub_eig, LinearAlgebra.eigvals(S)),
            S,
            state.kernel
        )
    end

    new_state = RAMState(trans, state.varinfo, S, step + 1, state.logjoint_internal)
    return new_state
end


# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::RAM_Sampler) = true 

Turing.Inference.init_strategy(::RAM_Sampler) = DynamicPPL.InitFromPrior()

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:RAMState}
    return state.varinfo
end


function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::RAM_Sampler,
    state::RAMState,
    params::AbstractVarInfo,
)
    # Update the vi
    #vi = state.varinfo
    #vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, params)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = Turing.Inference.Transition(model, vi, nothing)
    new_state = RAMState(new_transition, vi, state.kernel, state.step, lp)
    
    # Update the VarInfo with the new log density
    return new_state
end


# Write a function which proposes a new set of parameters
"""
    propose_RAM(rng::T_rng, vi::AbstractVarInfo, model::DynamicPPL.Model, spl::RAM_Sampler, S::T) where {T_rng <: AbstractRNG, T <: Cholesky}

Propose a new set of parameters for the RAM sampler.

# Arguments
- `rng::T_rng`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
- `S::T`: The Cholesky decomposition of the covariance matrix.
"""
function propose_RAM(
    rng::T_rng,
    vi::AbstractVarInfo,
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
    state::RAMState,
    S::T
) where {T_rng <: AbstractRNG, T <: Cholesky}
    # Grab the parameters we are performing inference on
    θ = vi[:]

    # Propose a new set of parameters
    proposal, U = proposewalkram(rng, θ, S.L)
    # Set the new parameters in the VarInfo
    proposal_vi = DynamicPPL.unflatten(vi, proposal)

    # Call a function to update the varinfo and evaluate the logdensity
    return Turing.Inference.setparams_varinfo!!(model, spl, state, proposal_vi), U
end

# Write a function which initialises the chain where there are warmup steps
"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::DynamicPPL.Model, spl::RAM_Sampler; initial_params=nothing, kwargs...)

Perform an initial warmup step for the RAM sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler;
    initial_params = Turing.Inference.init_strategy(spl),
    kwargs...,
)
    vis = DynamicPPL.VarInfo(model)
    _, vis = DynamicPPL.init!!(rng, model, vis, initial_params)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    # vis = DynamicPPL.default_varinfo(rng, model, spl)

    # # If there are initial parameters, set them and the logdensity
    # if initial_params !== nothing
    #     vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
    #     vis = last(DynamicPPL.evaluate!!(model, vis))
    # end

    # Link the variables
    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    S = cholesky(0.01 .* abs.(diagm(theta)) + (1e-6 * I))

    # get the old state
    orig_state = RAMState(prev_transition, vis, S, 1, lp_old)

    # Propose a new set of parameters and grab the log density
    new_state, _ = propose_RAM(rng, vis, model, spl, orig_state, S)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, orig_state
    end

    return new_state.transition, new_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::DynamicPPL.Model, spl::RAM_Sampler, state::RAMState; initial_params=nothing, kwargs...)

Perform a warmup step for the RAM sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
- `state::RAMState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
    state::RAMState;
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Get the kernel
    S = state.kernel

    # Propose the new parameters and get the log density
    proposal_state, _ = propose_RAM(rng, vis, model, spl, state, S)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return state.transition, state
    end

    # Create a new transition and state object if accepted
    # trans = Turing.Inference.Transition(vis, lp_proposal, true)
    # new_state = RAMState(trans, S, state.step)

    return proposal_state.transition, proposal_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::RAM_Sampler; initial_params=nothing, kwargs...)

Perform an initial sampling step for the RAM sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler;
    initial_params = Turing.Inference.init_strategy(spl),
    kwargs...,
)
    vis = DynamicPPL.VarInfo(model)
    _, vis = DynamicPPL.init!!(rng, model, vis, initial_params)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    # vis = DynamicPPL.default_varinfo(rng, model, spl)

    # # If there are initial parameters, set them and the logdensity
    # if initial_params !== nothing
    #     vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
    #     vis = last(DynamicPPL.evaluate!!(model, vis))
    # end    # Link the variables

    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)

    # Create a transition object in case of rejection
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    S = cholesky(0.01 .* abs.(diagm(theta)) + (1e-6 * I))

    # Get the old state
    orig_state = RAMState(prev_transition, vis, S, 1, lp_old)

    # Propose a new set of parameters and grab the log density
    proposal_state, U = propose_RAM(rng, vis, model, spl, orig_state, S)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : prev_transition

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    upd_state = update_kernel(accep ? proposal_state : orig_state, new_trans, U, spl, prob)

    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::RAM_Sampler, state::RAMState; initial_params=nothing, kwargs...)

Perform a sampling step for the RAM sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
- `state::RAMState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
    state::RAMState;
    kwargs...,
)
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Propose the new parameters and grab the log density
    proposal_state, U = propose_RAM(rng, vis, model, spl, state, state.kernel)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : state.transition

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    new_state = update_kernel(accep ? proposal_state : state, new_trans, U, spl, prob)

    return new_trans, new_state
end

# Define assume functions for the RAM sampler
#function DynamicPPL.assume(
#    rng::Random.AbstractRNG, spl::RAM_Sampler, dist::Distribution, vn::DynamicPPL.VarName, vi
#)
#    # DynamicPPL.updategid!(vi, vn, spl)
#    return DynamicPPL.assume(dist, vn, vi)
#end

# Define function to get log evidence
getlogevidence(transitions, sampler::RAM_Sampler, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:Turing.Inference.Transition}, model::DynamicPPL.Model, spl::RAM_Sampler, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the RAM sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:Turing.Inference.Transition}`: The Vector of transitions.
- `model::DynamicPPL.Model`: The model.
- `spl::RAM_Sampler`: The RAM sampler.
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
    model::DynamicPPL.Model,
    spl::RAM_Sampler,
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
