# write functions step and step_warmup which are used in the RAM_sampler.
# Using the algorithm constructed in https://arxiv.org/abs/1011.4381

# Start by defining the sampler

struct RAM_Sampler{space, T_accep <: Real, T_conv_step <: Real, T_lb_eig <: Real, T_ub_eig <: Real} <: AbstractMCMC.AbstractSampler
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
- `space::Tuple`: The parameter space for the sampler.
- `lb_eig::T3`: The lower bound for the eigenvalues for the acceptance of the kernel (default: -Inf).
- `ub_eig::T4`: The upper bound for the eigenvalues for the acceptance of the kernel (default: Inf).
- `bound_eigs::Bool`: Whether to bound the eigenvalues (default: false).
"""
function RAM_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Tuple;
    lb_eig::T3 = -Inf,
    ub_eig::T4 = Inf,
    bound_eigs::Bool = false,
) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}
    return RAM_Sampler{space, T1, T2, T3, T4}(acceptance_rate, convergence_step, lb_eig, ub_eig, bound_eigs)
end

function RAM_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Symbol...;
    lb_eig::T3 = -Inf,
    ub_eig::T4 = Inf,
    bound_eigs::Bool = false,
)  where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}
    return RAM_Sampler(acceptance_rate, convergence_step, space; lb_eig = lb_eig, ub_eig = ub_eig, bound_eigs = bound_eigs)
end

DynamicPPL.getspace(::RAM_Sampler{space, <: Any, <: Any, <: Any, <: Any}) where {space} = space
DynamicPPL.default_chain_type(sampler::Sampler{<:RAM_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
struct RAMState{
    TTrans<:AdvancedMH.Transition,
    TKernel<:Cholesky,
    TStep<:Integer,
}
    transition::TTrans
    kernel::TKernel
    step::TStep
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
    kernel::T_kernel
) where {T_trans <: AdvancedMH.AbstractTransition, T_kernel <: Cholesky}
    # Initialize step
    step = 1
    RAMState{T_trans, T_kernel, Int}(transition, kernel, step)
end

# Write a function which updates the kernel
"""
    update_kernel(state::RAMState, trans::AdvancedMH.Transition, U::T, spl::Sampler{<:RAM_Sampler}, prob::T3) where {T <: Vector, T3 <: Real}

Update the kernel matrix of the RAM sampler state based on the new transition and acceptance probability.

# Arguments
- `state::RAMState`: The current state of the sampler.
- `trans::AdvancedMH.Transition`: The new transition object.
- `U::T`: The proposed update vector.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::RAMState, trans::AdvancedMH.Transition, U::T, spl::Sampler{<:RAM_Sampler}, prob::T3) where {T <: Vector, T3 <: Real}
    # Grab useful quantities from the state
    S = state.kernel
    step = state.step

    # Follow maths from the paper to update the kernel
    # Do this via the Cholesky update and downdate to simplify calculations
    η = (step + 1.0)^(-spl.alg.convergence_step)
    accepdiff = (prob - spl.alg.acceptance_rate)
    β = sqrt(η * abs(accepdiff)) * S.L * U / norm(U)

    S = if sign(accepdiff) == 1
        LinearAlgebra.lowrankupdate(S, β)
    else
        LinearAlgebra.lowrankdowndate(S, β)
    end

    # Bound the eigenvalues if required
    if spl.alg.bound_eigs
        S = ifelse(
            all(x -> spl.alg.lb_eig <= x <= spl.alg.ub_eig, LinearAlgebra.eigvals(S)),
            S,
            state.kernel
        )
    end

    new_state = RAMState(trans, S, step + 1)
    return new_state
end

# Write a function which sets new parameters for varinfo and sets the new log density
"""
    set_params_ram(rng::T_rng, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:RAM_Sampler}, params::Vector{T}) where {T <: Real, T_rng <: AbstractRNG}

Set the parameters of the RAM sampler.

# Arguments
- `rng::T_rng`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `params::Vector{T}`: The new parameters.
"""
function set_params_ram(
    rng::T_rng,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:RAM_Sampler},
    params::Vector{T}
) where {T <: Real, T_rng <: AbstractRNG}
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
    propose_RAM(rng::T_rng, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:RAM_Sampler}, S::T) where {T_rng <: AbstractRNG, T <: Cholesky}

Propose a new set of parameters for the RAM sampler.

# Arguments
- `rng::T_rng`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `S::T`: The Cholesky decomposition of the covariance matrix.
"""
function propose_RAM(
    rng::T_rng,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:RAM_Sampler},
    S::T
) where {T_rng <: AbstractRNG, T <: Cholesky}
    # Grab the parameters we are performing inference on
    θ = vi[spl]

    # Propose a new set of parameters
    proposal, U = proposewalkram(rng, θ, S.L)

    # Call a function to update the varinfo and evaluate the logdensity
    return set_params_ram(rng, vi, model, spl, proposal), U
end

# Write a function which initialises the chain where there are warmup steps
"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:RAM_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the RAM sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:RAM_Sampler};
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
        vis = set_params_ram(rng, vis, model, spl, initial_params)
    end

    # Link the variables
    DynamicPPL.link!!(vis, spl, model)

    # Get the log density of the current parameters
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[spl]
    S = cholesky(0.01 .* abs.(diagm(theta)) + (1e-6 * I))

    # Propose a new set of parameters and grab the log density
    vis, _ = propose_RAM(rng, vis, model, spl, S)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, init_RAMState(prev_transition, S)
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    state = init_RAMState(trans, S)

    return trans, state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:RAM_Sampler}, state::RAMState; initial_params=nothing, kwargs...)

Perform a warmup step for the RAM sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `state::RAMState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:RAM_Sampler},
    state::RAMState;
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.transition.params
    lp_old = getlogp(vis)

    # Create a transition object (in case of rejection)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the kernel
    S = state.kernel

    # Propose the new parameters and get the log density
    vis, _ = propose_RAM(rng, vis, model, spl, S)
    lp_proposal = getlogp(vis)

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        new_state = RAMState(prev_transition, S, state.step)
        return prev_transition, state
    end

    # Create a new transition and state object if accepted
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    new_state = RAMState(trans, S, state.step)

    return trans, new_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:RAM_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the RAM sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:RAM_Sampler};
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
        vis = set_params_ram(rng, vis, model, spl, initial_params)
    end

    # Link the variables
    DynamicPPL.link!!(vis, spl, model)

    # Get the log density of the current parameters
    lp_old = getlogp(vis)

    # Create a transition object in case of rejection
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[spl]
    S = cholesky(0.01 .* abs.(diagm(theta)) + (1e-6 * I))

    # Propose a new set of parameters and grab the log density
    vis, U = propose_RAM(rng, vis, model, spl, S)
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
    new_state = init_RAMState(new_trans, S)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    upd_state = update_kernel(new_state, new_trans, U, spl, prob)

    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:RAM_Sampler}, state::RAMState; initial_params=nothing, kwargs...)

Perform a sampling step for the RAM sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:RAM_Sampler}`: The RAM sampler.
- `state::RAMState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:RAM_Sampler},
    state::RAMState;
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = state.transition.params
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    # Propose the new parameters and grab the log density
    vis, U = propose_RAM(rng, vis, model, spl, state.kernel)
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
    new_state = update_kernel(state, new_trans, U, spl, prob)

    return new_trans, new_state
end

# Define assume functions for the RAM sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Turing.Sampler{<:RAM_Sampler}, dist::Distribution, vn::Turing.VarName, vi
)
    DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end

function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:RAM_Sampler},
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
    spl::Turing.Sampler{<:RAM_Sampler},
    dists::Union{Distribution, AbstractArray{<:Distribution}},
    vns::AbstractArray{<:Turing.VarName},
    var::AbstractArray,
    vi,
)
    # Just defer to `SampleFromPrior`.
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dists, var, vns, vi)
end

# Define observe functions for the RAM sampler
function DynamicPPL.observe(spl::Turing.Sampler{<:RAM_Sampler}, d::Distribution, value, vi)
    return DynamicPPL.observe(d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Turing.Sampler{<:RAM_Sampler},
    ds::Union{Distribution, AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(ds, value, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::Turing.Sampler{<:RAM_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:AdvancedMH.AbstractTransition}, model::AbstractMCMC.AbstractModel, spl::Turing.Sampler{<:RAM_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the RAM sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:AdvancedMH.AbstractTransition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Turing.Sampler{<:RAM_Sampler}`: The RAM sampler.
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
    spl::Turing.Sampler{<:RAM_Sampler},
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
