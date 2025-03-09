#  write a functions step and step_warmup which are used in the DRAMGS_sampler.
# Based on Algorithm from https://arxiv.org/pdf/2010.04190

#TODO: Test which probability should be used to update the scaling parameter (maybe after initial sample???)

# Start by defining the sampler

struct DRAMGS_Sampler{space, T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_max_steps <: Int, T_rej_steps <: AbstractVector{<:Real}} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Bool
    noise_scaling::T_scale
    max_delayed_rejection_steps::T_max_steps
    rejection_step_sizes::T_rej_steps
end

function DRAMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Tuple;
    noise_stabiliser::Bool=false,
    noise_stabiliser_scaling::T3=1e-6,
    max_delayed_rejection_steps::T4=3,
    rejection_step_sizes::T5=fill(1.0, T4)
) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Int, T5 <: AbstractVector{<:Real}}
    return DRAMGS_Sampler{space, T1, T2, T3, T4, T5}(acceptance_rate, convergence_step, noise_stabiliser, noise_stabiliser_scaling, max_delayed_rejection_steps, rejection_step_sizes)
end

"""
    DRAMGS_Sampler(acceptance_rate::T1, convergence_step::T2, space::Tuple) where {T1 <: Real, T2 <: Real}

Create a new DRAMGS sampler with the specified acceptance rate, convergence step, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `convergence_step::T2`: A tuning parameter for the rate of convergence/step size of the sampler.
- `space::Tuple`: The parameter space for the sampler.
"""
function DRAMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Symbol...;
    kwargs...
) where {T1 <: Real, T2 <: Real}
    return DRAMGS_Sampler(acceptance_rate, convergence_step, space; kwargs...)
end

"""
    DRAMGS_Sampler(acceptance_rate::T1, convergence_step::T2, noise_stabiliser::Bool, noise_stabiliser_scaling::T3, space::Tuple) where {T1 <: Real, T2 <: Real, T3 <: Real}

Create a new DRAMGS sampler with the specified acceptance rate, convergence step, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `convergence_step::T2`: A tuning parameter for the rate of convergence/step size of the sampler.
- `noise_stabiliser::Bool`: Whether to use a noise stabiliser.
- `noise_stabiliser_scaling::T3`: The scaling factor for the noise stabiliser.
- `space::Tuple`: The parameter space for the sampler.
"""
function DRAMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    noise_stabiliser::Bool,
    noise_stabiliser_scaling::T3,
    space::Symbol...;
) where {T1 <: Real, T2 <: Real, T3 <: Real}
    return DRAMGS_Sampler(acceptance_rate, convergence_step, space; noise_stabiliser = noise_stabiliser, noise_stabiliser_scaling = noise_stabiliser_scaling)
end


DynamicPPL.getspace(::DRAMGS_Sampler{space, <: Any, <: Any, <: Any, <: Any, <: Any}) where {space} = space

DynamicPPL.default_chain_type(sampler::Sampler{<:DRAMGS_Sampler}) = MCMCChains.Chains

# Write a function which updates the kernel
"""
    update_kernel(state::AMGSState, trans::T, spl::Sampler{<:DRAMGS_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}

Update the kernel matrix, and internal quantities of the AMGS sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::AMGSState, trans::T, spl::Sampler{<:DRAMGS_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}
    # Grab useful quantities from the state
    Σ = state.kernel
    μ = state.μ
    step = state.step

    # Follow maths from arXiv paper to update the kernel and the global scale β
    β = state.β
    η = (step + 1.0)^(-spl.alg.convergence_step)
    new_β = β + η * (prob - spl.alg.acceptance_rate)
    Δ = trans.params[:] - μ
    new_μ = (1.0 - η) * μ + (η * trans.params[:])
    
    # Add noise to the kernel if requested
    noise_term = spl.alg.use_noise_stabiliser * spl.alg.noise_scaling * η * rand(size(Δ)[1])
    new_Σ = (1.0 - η) * Σ + (η * (Δ * (Δ'))) + diagm(noise_term)
    new_state = AMGSState(trans, new_μ, new_Σ, step + 1, new_β)
    return new_state
end

# Function to set the new parameters
"""
    set_params_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:DRAMGS_Sampler}, params::Vector{T}) where {T <: Real}

Set the parameters of the DRAMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `params::Vector{T}`: The new parameters.
"""
function set_params_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:DRAMGS_Sampler},
    params::Vector{T}
) where {T <: Real}
    # Create function to evaluate the model to get the log density using the right context
    new_ldf = Base.Fix1(
        Turing.LogDensityProblems.logdensity,
        Turing.LogDensityFunction(
            model,
            vi,
            DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context))
        )
    )

    # Evaluate the logdensity for the new params
    lp_init = new_ldf(params)

    # Update the VarInfo object with the new parameters
    vi = DynamicPPL.unflatten(vi,
           DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
           params)

    # Update the VarInfo with the new log density
    return setlogp!!(vi, lp_init)
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::DRAMGS_Sampler) = true 

# drop space function
Turing.Inference.drop_space(alg::DRAMGS_Sampler{space,T1,T2,T3,T4,T5}) where {space,T1,T2,T3,T4,T5} = DRAMGS_Sampler{(),T1,T2,T3,T4,T5}(alg.acceptance_rate, alg.convergence_step, alg.use_noise_stabiliser, alg.noise_scaling, alg.max_delayed_rejection_steps, alg.rejection_step_sizes)

# Get varinfo
function Turing.Inference.varinfo(state::T) where {T<:AMGSState}
    # display(state.transition.params)
    # display(typeof(state.transition.params))
    return state.transition.params
end
# function isgibbscomponent(::T) where {T<:AMGS_Sampler} 
#     return true
# end


function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:DRAMGS_Sampler},
    state::AMGSState,
    params::AbstractVarInfo,
)

    # Update the vi
    vi  = last(DynamicPPL.evaluate!!(model, params, DynamicPPL.leafcontext(model.context)))

    #update the model state
    new_transition = AdvancedMH.Transition(vi, getlogp(vi), state.transition.accepted)
    new_state = AMGSState(new_transition, state.μ, state.kernel, state.step, state.β)
    
    # Update the VarInfo with the new log density
    return new_state
end



# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:DRAMGS_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the DRAMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:DRAMGS_Sampler},
    Σ::AbstractMatrix
)
    # Grab the parameters we are performing inference on
    θ = vi[:]

    # Propose a new set of parameters
    proposal = proposewalk(rng, θ, Σ)

    # Call a function to update the varinfo and evaluate the logdensity
    return set_params_amgs(rng, vi, model, spl, proposal) 
end

# Write a function which initialises the chain where there are warmup steps
"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:DRAMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the DRAMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:DRAMGS_Sampler};
    initial_params=nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = DynamicPPL.default_varinfo(rng, model, spl)

    # If there are initial parameters, set them and the logdensity
    if initial_params !== nothing
        vis = set_params_amgs(rng, vis, model, spl, initial_params)
    end

    # Link the variables
    DynamicPPL.link!!(vis, spl, model)

    # Get the log density of the current parameters
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    # Propose a new set of parameters and grab the log density
    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if accep

        # Create a new transition and state object if accepted
        trans = AdvancedMH.Transition(vis, lp_proposal, true)
        state = init_AMGSState(trans, theta, Σ)

        return trans, state
    end

    # Check whether to perform a delayed resample step
    if spl.alg.max_delayed_rejection_steps > 0
            largest_last_lp = getlogp(vis)
            for i in 1:spl.alg.max_delayed_rejection_steps
                prev_lp = getlogp(vis)
                largest_last_lp = prev_lp > largest_last_lp ? prev_lp : largest_last_lp

            # Propose the new parameters and get the log density
            vis = propose_amgs(rng, vis, model, spl, spl.alg.rejection_step_sizes[i] * Σ)
            lp_proposal = getlogp(vis)

            # Check whether to accept the new parameters and return the previous transition if not
            _, accep = mh_dr_accept_logdens(rng, lp_proposal, largest_last_lp, lp_old)
            if accep
                # Create a new transition and state object if accepted
                trans = AdvancedMH.Transition(vis, lp_proposal, true)
                state = init_AMGSState(trans, theta, Σ)

                return trans, state
            end
        end
    end

    return prev_transition, init_AMGSState(prev_transition, theta, Σ)

end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:DRAMGS_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the DRAMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:DRAMGS_Sampler},
    state::AMGSState;
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.transition.params
    lp_old = getlogp(vis)

    # Create a transition object (in case of rejection)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the kernel
    Σ = state.kernel

    # Propose the new parameters and get the log density
    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters and return
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if accep

        # Create a new transition and state object if accepted
        trans = AdvancedMH.Transition(vis, lp_proposal, true)
        state = AMGSState(trans, state.μ, Σ, state.step, state.β)

        return trans, state
    end

    # Check whether to perform a delayed resample step
    if spl.alg.max_delayed_rejection_steps > 0
            largest_last_lp = getlogp(vis)
            for i in 1:spl.alg.max_delayed_rejection_steps
                prev_lp = getlogp(vis)
                largest_last_lp = prev_lp > largest_last_lp ? prev_lp : largest_last_lp

            # Propose the new parameters and get the log density
            vis = propose_amgs(rng, vis, model, spl, spl.alg.rejection_step_sizes[i] * Σ)
            lp_proposal = getlogp(vis)

            # Check whether to accept the new parameters 
            _, accep = mh_dr_accept_logdens(rng, lp_proposal, largest_last_lp, lp_old)
            if accep
                # Create a new transition and state object if accepted
                trans = AdvancedMH.Transition(vis, lp_proposal, true)
                state = AMGSState(trans, state.μ, Σ, state.step, state.β)

                return trans, state
            end
        end
    end

    return prev_transition, AMGSState(prev_transition, state.μ, Σ, state.step, state.β)
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:DRAMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the DRAMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:DRAMGS_Sampler};
    initial_params=nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = DynamicPPL.default_varinfo(rng, model, spl)

    # If there are initial parameters, set them and the logdensity
    if initial_params !== nothing
        vis = set_params_amgs(rng, vis, model, spl, initial_params)
    end

    # Link the variables
    DynamicPPL.link!!(vis, spl, model)

    # Get the log density of the current parameters
    lp_old = getlogp(vis)

    # Create a transition object in case of rejection
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    # Propose a new set of parameters and grab the log density
    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters 
    _, accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    if !accep
        new_trans = prev_transition

        # Check whether to perform a delayed resample step
        if spl.alg.max_delayed_rejection_steps > 0
            largest_last_lp = getlogp(vis)
            for i in 1:spl.alg.max_delayed_rejection_steps
                prev_lp = getlogp(vis)
                largest_last_lp = prev_lp > largest_last_lp ? prev_lp : largest_last_lp

                # Propose the new parameters and get the log density
                vis = propose_amgs(rng, vis, model, spl, spl.alg.rejection_step_sizes[i] * Σ)
                lp_proposal = getlogp(vis)

                # Check whether to accept the new parameters 
                accep = mh_dr_accept_logdens(rng, lp_proposal, largest_last_lp, lp_old)
                if accep
                    # Create a new transition 
                    new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
                    break
                end
            end
        end
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end

    # Construct the initial state of the sampler
    new_state = init_AMGSState(new_trans, new_trans.params[:], Σ)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    upd_state = update_kernel(new_state, new_trans, spl, prob)

    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:DRAMGS_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the DRAMGS sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:DRAMGS_Sampler},
    state::AMGSState;
    kwargs...
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = state.transition.params
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Propose the new parameters and grab the log density
    vis = propose_amgs(rng, vis, model, spl, exp(state.β) * state.kernel)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    prob = exp(min(0.0, (lp_proposal - lp_old)))

    # Update the kernel and return the new transition and state
    # prob = exp(min(0.0, (lp_proposal - lp_old)))

    # Construct the transition object to be returned
    if !accep
        new_trans = prev_transition

        # Check whether to perform a delayed resample step
        if spl.alg.max_delayed_rejection_steps > 0
            largest_last_lp = getlogp(vis)
            for i in 1:spl.alg.max_delayed_rejection_steps
                prev_lp = getlogp(vis)
                largest_last_lp = prev_lp > largest_last_lp ? prev_lp : largest_last_lp

                # Propose the new parameters and get the log density
                vis = propose_amgs(rng, vis, model, spl, spl.alg.rejection_step_sizes[i] * exp(state.β) * state.kernel)
                lp_proposal = getlogp(vis)

                # Check whether to accept the new parameters 
                _, accep = mh_dr_accept_logdens(rng, lp_proposal, largest_last_lp, lp_old)
                if accep
                    # Create a new transition 
                    new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
                    break
                end
            end
        end
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end
    
    new_state = update_kernel(state, new_trans, spl, prob)

    return new_trans, new_state
end


# Define assume functions for the DRAMGS sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Turing.Sampler{<:DRAMGS_Sampler}, dist::Distribution, vn::Turing.VarName, vi
)
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end


function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:DRAMGS_Sampler},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:Turing.VarName},
    var::AbstractMatrix,
    vi,
)
    # DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dist, var, vns, vi)
end

function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:DRAMGS_Sampler},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:Turing.VarName},
    var::AbstractArray,
    vi,
)
    # Just defer to `SampleFromPrior`.
    # DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dists, var, vns, vi)
end

# Define observe functions for the DRAMGS sampler
function DynamicPPL.observe(spl::Turing.Sampler{<:DRAMGS_Sampler}, d::Distribution, value, vi)
    return DynamicPPL.observe(d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Turing.Sampler{<:DRAMGS_Sampler},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi
)
    return DynamicPPL.dot_observe(ds, value, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::Turing.Sampler{<:DRAMGS_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:AdvancedMH.AbstractTransition}, model::AbstractMCMC.AbstractModel, spl::Turing.Sampler{<:DRAMGS_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the DRAMGS sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:AdvancedMH.AbstractTransition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Turing.Sampler{<:DRAMGS_Sampler}`: The DRAMGS sampler.
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
    ts::Vector{<:AdvancedMH.AbstractTransition},
    model::AbstractMCMC.AbstractModel,
    spl::Turing.Sampler{<:DRAMGS_Sampler},
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
    extra_params, extra_values = Turing.Inference.get_transition_extras(map(x -> x.params, ts))

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
