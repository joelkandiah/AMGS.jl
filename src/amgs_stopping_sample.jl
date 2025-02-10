# write a functions step and step_warmup which are used in the AMGS_Stopping_sampler.
# Write code based on algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y combined with the extensions in the appendix of https://arxiv.org/abs/2408.04178 (note appendix to be published soon)

# Start by defining the sampler
# Note added the stopping conditions as parameters to the sampler
# The max number of steps where adaptation occurs, the lower and upper bounds of the acceptance rate to check at each average interval
struct AMGS_Stopping_Sampler{space, T_accep <: Real, T_conv_step <: Real, T_adapt_period <: Integer, T_accep_l <: Real, T_accep_u <: Real, T_accep_interval <: Integer} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    max_adapt_period::T_adapt_period
    lower_stopping_accep::T_accep_l
    upper_stopping_accep::T_accep_u
    average_interval::T_accep_interval
end

"""
    AMGS_Stopping_Sampler(acceptance_rate::T1, convergence_step::T2, max_adapt_period::T3, lower_stopping_accep::T4, upper_stopping_accep::T5, average_interval::T6, space::Tuple) where {T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer}

Create a new AMGS Stopping sampler with the specified acceptance rate, convergence step, maximum adaptation period, lower and upper stopping acceptance rates, average interval, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `convergence_step::T2`: A tuning parameter for the rate of convergence/step size of the sampler.
- `max_adapt_period::T3`: The maximum number of steps where adaptation occurs.
- `lower_stopping_accep::T4`: The lower bound of the acceptance rate to check at each average interval.
- `upper_stopping_accep::T5`: The upper bound of the acceptance rate to check at each average interval.
- `average_interval::T6`: The number of steps over which to average the acceptance rate.
- `space::Tuple`: The parameter space for the sampler.
"""
function AMGS_Stopping_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    max_adapt_period::T3,
    lower_stopping_accep::T4,
    upper_stopping_accep::T5,
    average_interval::T6,
    space::Tuple;
) where {T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer}
    return AMGS_Stopping_Sampler{space, T1, T2, T3, T4, T5, T6}(acceptance_rate, convergence_step, max_adapt_period, lower_stopping_accep, upper_stopping_accep, average_interval)
end

function AMGS_Stopping_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    max_adapt_period::T3,
    lower_stopping_accep::T4,
    upper_stopping_accep::T5,
    average_interval::T6,
    space::Symbol...;
) where {T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real, T5 <: Real, T6 <: Integer}
    return AMGS_Stopping_Sampler(acceptance_rate, convergence_step, max_adapt_period, lower_stopping_accep, upper_stopping_accep, average_interval, space;)
end

DynamicPPL.getspace(::AMGS_Stopping_Sampler{space, <: Any, <: Any, <: Any, <: Any, <: Any}) where {space} = space
DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_Stopping_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
# In addition to the AMGSState, we also store the previous accepts as well as whether to adaptation has ended
struct AMGSStoppingState{
    TTrans<:AdvancedMH.Transition,
    Tμ<:Vector,
    TKernel<:Matrix,
    TStep<:Integer,
    Tβ<:Real,
    TΔi<:Vector
}
    transition::TTrans
    μ::Tμ
    kernel::TKernel
    step::TStep
    β::Tβ
    Δi::TΔi
    stopped::Bool
end

# Define the initial state of the sampler
"""
    init_AMGSStoppingState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel, sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS Stopping sampler with the given transition, mean vector μ, kernel matrix, and sampler.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
- `sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
"""
function init_AMGSStoppingState(
    transition::T_trans,
    μ::Vector{T_μ},
    kernel::T_kernel,
    sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}
) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    Δi = zeros(eltype(μ), sampler.alg.average_interval)

    AMGSStoppingState{T_trans, Vector{T_μ}, T_kernel, Int, typeof(β), typeof(Δi)}(transition, μ, kernel, step, β, Δi, false)
end

# Define the initial state of the sampler in a 1D case
"""
    init_AMGSStoppingState(transition::T_trans, μ::T_μ, kernel::T_kernel, sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS Stopping sampler with the given transition, mean vector μ, kernel matrix, and sampler for a 1D case.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean value.
- `kernel::T_kernel`: The initial kernel matrix.
- `sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
"""
function init_AMGSStoppingState(
    transition::T_trans,
    μ::T_μ,
    kernel::T_kernel,
    sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}
) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    μ = [μ]
    AMGSStoppingState{T_trans, T_μ, T_kernel, Int, typeof(β)}(transition, μ, kernel, step, β, sampler, false)
end

# Write a function which updates the kernel
"""
    update_kernel(state::AMGSStoppingState, trans::T, spl::Sampler{<:AMGS_Stopping_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}

Update the kernel matrix, and internal quantities of the AMGS Stopping sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSStoppingState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::AMGSStoppingState, trans::T, spl::Sampler{<:AMGS_Stopping_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}
    # Grab useful quantities from the state
    Σ = state.kernel
    μ = state.μ
    step = state.step
    β = state.β

    # Follow maths from arXiv paper to update the kernel and the global scale β
    η = (step + 1.0)^(-spl.alg.convergence_step)
    β = β + η * (prob - spl.alg.acceptance_rate)
    Δ = trans.params[spl] - μ
    μ = (1.0 - η) * μ + (η * trans.params[spl])
    Σ = (1.0 - η) * Σ + (η * (Δ * (Δ')))
    new_state = AMGSStoppingState(trans, μ, Σ, step + 1, β, state.Δi, state.stopped)
    return new_state
end

# Function to set the new parameters
"""
    set_params_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Sampler}, params::Vector{T}) where {T <: Real}

Set the parameters of the AMGS Stopping sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `params::Vector{T}`: The new parameters.
"""
function set_params_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler},
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
    lp_proposal = new_ldf(params)

    # Update the VarInfo object with the new parameters
    vi = DynamicPPL.unflatten(vi,
        DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
        params)

    # Update the VarInfo with the new log density
    return setlogp!!(vi, lp_proposal)
end

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGS Stopping sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Stopping_Sampler},
    Σ::AbstractMatrix
)
    # Grab the parameters we are performing inference on
    θ = vi[spl]

    # Propose a new set of parameters
    proposal = proposewalk(rng, θ, Σ)

    # Call a function to update the varinfo and evaluate the logdensity
    return set_params_amgs(rng, vi, model, spl, proposal)
end

# Write a function which initialises the chain where there are warmup steps
"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Stopping_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGS Stopping sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_Stopping_Sampler}`: The AMGS Stopping sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
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
    theta = vis[spl]
    Σ = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    # Propose a new set of parameters and grab the log density
    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, init_AMGSStoppingState(prev_transition, theta, Σ, spl)
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    state = init_AMGSStoppingState(trans, theta, Σ, spl)

    return trans, state
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
        new_state = AMGSStoppingState(prev_transition, state.μ, Σ, state.step, state.β, state.Δi, state.stopped)
        return prev_transition, state
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    new_state = AMGSStoppingState(trans, state.μ, Σ, state.step, state.β, state.Δi, state.stopped)

    return trans, new_state
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
    theta = vis[spl]
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
    new_state = init_AMGSStoppingState(new_trans, theta, Σ, spl)

    # Update Δi if accepted and there is a non-zero averaging interval
    if accep && spl.alg.average_interval >= 1
        # (Want to update only the Δi[1] element as we can guarantee adaptation has started as long as the averaging interval is greater than 0)
        new_state_Δi = copy(new_state.Δi)
        new_state_Δi[1] = 1
        new_state = AMGSStoppingState(new_state.transition, new_state.μ, new_state.kernel, new_state.step, new_state.β, new_state_Δi, new_state.stopped)
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
    vis = state.transition.params
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the parameters of interest
    theta = vis[spl]

    # Propose the new parameters and get the log density
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
    new_state = AMGSStoppingState(new_trans, state.μ, state.kernel, state.step, state.β, upd_Δi, stopped)

    # Update the kernel if adaptation has not stopped
    if !new_state.stopped
        new_state = update_kernel(new_state, new_trans, spl, prob)
    end

    return new_trans, new_state
end

function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Turing.Sampler{<:AMGS_Stopping_Sampler}, dist::Distribution, vn::T_VN, vi
) where {T_VN <: Turing.VarName}
    DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end


function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:AMGS_Stopping_Sampler},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:Turing.VarName},
    var::AbstractMatrix,
    vi,
)
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dist, var, vns, vi)
end

function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:AMGS_Stopping_Sampler},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:Turing.VarName},
    var::AbstractArray,
    vi,
)
    # Just defer to `SampleFromPrior`.
   DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dists, var, vns, vi)
end

function DynamicPPL.observe(spl::Turing.Sampler{<:AMGS_Stopping_Sampler}, d::Distribution, value, vi)
    return DynamicPPL.observe(d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Turing.Sampler{<:AMGS_Stopping_Sampler},
    ds::Union{Distribution,AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(ds, value, vi)
end

getlogevidence(transitions, sampler::Turing.Sampler{<:AMGS_Stopping_Sampler}, state) = missing


function AbstractMCMC.bundle_samples(
    ts::Vector{<:AdvancedMH.AbstractTransition},
    model::AbstractMCMC.AbstractModel,
    spl::Turing.Sampler{<:AMGS_Stopping_Sampler},
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