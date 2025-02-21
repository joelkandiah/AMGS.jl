# write a functions step and step_warmup which are used in the AMGS_sampler.
# Based on Algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y

# Start by defining the sampler

struct AMGS_Sampler{space, T_accep <: Real, T_conv_step <: Real} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
end

function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Tuple;
) where {T1 <: Real, T2 <: Real}
    return AMGS_Sampler{space, T1, T2}(acceptance_rate, convergence_step)
end


"""
    AMGS_Sampler(acceptance_rate::T1, convergence_step::T2, space::Tuple) where {T1 <: Real, T2 <: Real}

Create a new AMGS sampler with the specified acceptance rate, convergence step, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `convergence_step::T2`: A tuning parameter for the rate of convergence/step size of the sampler.
- `space::Tuple`: The parameter space for the sampler.
"""
function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Symbol...;
)  where {T1 <: Real, T2 <: Real}
    return AMGS_Sampler(acceptance_rate, convergence_step, space;)
end

DynamicPPL.getspace(::AMGS_Sampler{space, <: Any, <: Any}) where {space} = space

DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
struct AMGSState{
    TTrans<:AdvancedMH.Transition,
    Tμ<:Vector,
    TKernel<:Matrix,
    TStep<:Integer,
    Tβ<:Real
}
    transition::TTrans
    μ::Tμ
    kernel::TKernel
    step::TStep
    β::Tβ
end

# Define the initial state of the sampler
"""
    init_AMGSState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS sampler with the given transition, mean vector μ, and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_AMGSState(
    transition::T_trans,
    μ::Vector{T_μ},
    kernel::T_kernel
) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    AMGSState{T_trans, Vector{T_μ}, T_kernel, Int, typeof(β)}(transition, μ, kernel, step, β)
end

# Define the initial state of the sampler in a 1D case
function init_AMGSState(
    transition::T_trans,
    μ::T_μ,
    kernel::T_kernel
) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    μ = [μ]
    AMGSState{T_trans, T_μ, T_kernel, Int, typeof(β)}(transition, μ, kernel, step, β)
end

# Construct a transition object using our parameters of interest (varinfo instead of params)
AdvancedMH.Transition(
    params::T_params,
    lp::T_lp,
    accepted::Bool,
) where {T_params <: AbstractVarInfo, T_lp <: Real} = AdvancedMH.Transition{T_params, T_lp}(params, lp, accepted)

# Write a function which updates the kernel
"""
    update_kernel(state::AMGSState, trans::T, spl::Sampler{<:AMGS_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}

Update the kernel matrix, and internal quantities of the AMGS sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::AMGSState, trans::T, spl::Sampler{<:AMGS_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}
    # Grab useful quantities from the state
    Σ = state.kernel
    μ = state.μ
    step = state.step

    # Follow maths from arXiv paper to update the kernel and the global scale β
    β = state.β
    η = (step + 1.0)^(-spl.alg.convergence_step)
    β = β + η * (prob - spl.alg.acceptance_rate)
    Δ = trans.params[spl] - μ
    μ = (1.0 - η) * μ + (η * trans.params[spl])
    Σ = (1.0 - η) * Σ + (η * (Δ * (Δ')))
    new_state = AMGSState(trans, μ, Σ, step + 1, β)
    return new_state
end

# Function to set the new parameters
"""
    set_params_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Sampler}, params::Vector{T}) where {T <: Real}

Set the parameters of the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `params::Vector{T}`: The new parameters.
"""
function set_params_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
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

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
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
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler};
    initial_params=nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = VarInfo()
    spl_init = DynamicPPL.initialsampler(spl)
    model(rng, vis, spl_init)

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
    if !accep
        return prev_transition, init_AMGSState(prev_transition, theta, Σ)
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    state = init_AMGSState(trans, theta, Σ)

    return trans, state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
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

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        new_state = AMGSState(prev_transition, state.μ, Σ, state.step, state.β)
        return prev_transition, state
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    new_state = AMGSState(trans, state.μ, Σ, state.step, state.β)

    return trans, new_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_Sampler};
    initial_params=nothing,
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = VarInfo()
    spl_init = DynamicPPL.initialsampler(spl)
    model(rng, vis, spl_init)

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
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    if !accep
        new_trans = prev_transition
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end

    # Construct the initial state of the sampler
    new_state = init_AMGSState(new_trans, theta, Σ)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    upd_state = update_kernel(new_state, new_trans, spl, prob)

    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_Sampler}, state::AMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGS sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_Sampler}`: The AMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
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

    # Construct the transition object to be returned
    if !accep
        new_trans = prev_transition
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    new_state = update_kernel(state, new_trans, spl, prob)

    return new_trans, new_state
end


# Define assume functions for the AMGS sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Turing.Sampler{<:AMGS_Sampler}, dist::Distribution, vn::Turing.VarName, vi
)
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end


function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:AMGS_Sampler},
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
    spl::Turing.Sampler{<:AMGS_Sampler},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:Turing.VarName},
    var::AbstractArray,
    vi,
)
    # Just defer to `SampleFromPrior`.
    # DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dists, var, vns, vi)
end

# Define observe functions for the AMGS sampler
function DynamicPPL.observe(spl::Turing.Sampler{<:AMGS_Sampler}, d::Distribution, value, vi)
    return DynamicPPL.observe(d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Turing.Sampler{<:AMGS_Sampler},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(ds, value, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::Turing.Sampler{<:AMGS_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:AdvancedMH.AbstractTransition}, model::AbstractMCMC.AbstractModel, spl::Turing.Sampler{<:AMGS_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGS sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:AdvancedMH.AbstractTransition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Turing.Sampler{<:AMGS_Sampler}`: The AMGS sampler.
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
    spl::Turing.Sampler{<:AMGS_Sampler},
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
