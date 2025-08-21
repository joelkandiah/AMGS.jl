#  write a functions step and step_warmup which are used in the AMGS_sampler.
# Based on Algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y

# Start by defining the sampler

struct new_AMGS_Sampler{ T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_el_cov_mat } <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Bool
    noise_scaling::T_scale
    use_custom_init::Bool
    init_covariance::Matrix{T_el_cov_mat}
end

# Function to construct sampler without the inital covariance matrix
function new_AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3,
) where {T1 <: Real, T2 <: Real, T3 <: Real}
    return new_AMGS_Sampler{T1, T2, T3, Float64}(acceptance_rate, convergence_step, use_noise_stabiliser, noise_scaling, false, zeros(Float64, 0, 0))
end

function new_AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
) where {T1 <: Real, T2 <: Real}
    return new_AMGS_Sampler{T1, T2, Float64, Float64}(acceptance_rate, convergence_step, false, 1e-6, false, zeros(Float64,0,0))
end

DynamicPPL.default_chain_type(sampler::Sampler{<:new_AMGS_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
struct newAMGSState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    Tμ<:AbstractVector,
    TKernel<:Matrix,
    TStep<:Integer,
    Tβ<:Real,
    TL<:Real    
}
    transition::TTrans
    varinfo::TVI
    μ::Tμ
    kernel::TKernel
    step::TStep
    β::Tβ
    logjoint_internal::TL
end

# Define the initial state of the sampler
"""
    init_newAMGSState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS sampler with the given transition, mean vector μ, and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_newAMGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    newAMGSState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, μ, kernel, step, β, lp)
end

"""
    init_newAMGSState(transition::T_trans, μ::T_μ, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: Matrix}

Initialize the state of the AMGS sampler with the given transition, mean value μ, and kernel matrix for a 1D case.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean value.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_newAMGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    new_μ = [μ]
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    newAMGSState{T_trans, TVI, typeof(new_μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, new_μ, kernel, step, β, lp)
end

"""
_add_noise!(Δ², noise_scaling, θ_size, use_noise_stabiliser)

add noise to the kernel matrix Δ² based on the noise scaling and whether noise stabiliser is used.

# Arguments
- `Δ²`: The kernel matrix to which noise will be added.
- `noise_scaling`: The scaling factor for the noise.
- `θ_size`: The size of the parameter vector θ.
- `use_noise_stabiliser`: A boolean indicating whether to use noise stabilisation.
"""
# function _add_noise!(Δ², noise_scaling, θ_size, ::Val{true})
#    Δ² .= Δ² .+ diagm(noise_scaling .* rand(θ_size))
#end

#function _add_noise!(Δ², noise_scaling, θ_size, ::Val{false})
#    # Do nothing
#end

# Write a function which updates the kernel
"""
update_kernel(state::newAMGSState, trans::T, spl::Sampler{<:new_AMGS_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real}

Update the kernel matrix, and internal quantities of the AMGS sampler state based on the new transition and acceptance probability.

# Arguments
- `state::newAMGSState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::newAMGSState, trans::T, spl::Sampler{<:new_AMGS_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real}
 
     # Grab useful quantities from the state
     μ = state.μ
     step = state.step
     θ = state.varinfo[:]

     # Follow maths from arXiv paper to update the kernel and the global scale β
     η = (step + 1.0)^(-spl.alg.convergence_step)
     β = state.β + η * (prob - spl.alg.acceptance_rate)
     @inbounds @simd for i in eachindex(μ, θ)
         μ[i] = muladd(η, θ[i]-μ[i], μ[i])
     end
     # map!((x, y) -> ((1.0 - η) * x + (η * y)), μ, μ, θ)
     # @. μ = (1.0 - η) * μ + (η * θ)
     #new theta is actually Δ
     map!(-, θ, θ, μ)
     # Δ² = θ * transpose(θ)
     Δ² = similar(state.kernel)
     mul!(Δ², θ, θ')   # BLAS optimized, avoids allocations

     #new_Σ = copy(state.kernel)
     #BLAS.syrk!('U', 'N', η, θ, 1.0 - η, parent(new_Σ))
     
     # Add noise to the kernel if requested
     # _add_noise!(new_Σ, spl.alg.noise_scaling * η, size(θ)[1], Val(spl.alg.use_noise_stabiliser))
     _add_noise!(Δ², spl.alg.noise_scaling, size(θ)[1], Val(spl.alg.use_noise_stabiliser))
     # new_Σ = Symmetric(new_Σ, :U)
     # Δ² .= Δ² .+ diagm(spl.alg.use_noise_stabiliser .* spl.alg.noise_scaling .* rand(size(θ)[1]))
     new_Σ = @. (1.0 - η) * state.kernel + (η * Δ²) 
     return new_Σ, newAMGSState(trans, state.varinfo, μ, new_Σ, step + 1, β, state.logjoint_internal)
end


# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::new_AMGS_Sampler) = true 

# Get varinfo
function get_varinfo(state::T) where {T<:newAMGSState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:new_AMGS_Sampler},
    state::newAMGSState,
    params::AbstractVarInfo,
)

    # Update the vi
    vi = state.varinfo
    vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, vi)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = Turing.Inference.Transition(model, vi, nothing)
    new_state = newAMGSState(new_transition, vi, state.μ, state.kernel, state.step, state.β, lp) 
    
    # Update the VarInfo with the new log density
    return new_state
end

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:new_AMGS_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:new_AMGS_Sampler},
    state::newAMGSState,
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
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:new_AMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:new_AMGS_Sampler};
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
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    if(!spl.alg.use_custom_init)
        Σ = 0.01 .* diagm(abs.(theta)) + (1e-6 * I)
    else
        # If we are using a custom initialisation, we need to use the initial covariance matrix
        Σ = spl.alg.init_covariance
    end
    orig_state = init_newAMGSState(prev_transition, vis, theta, Σ)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, orig_state 
    end

    # Create a new transition and state object if accepted
    # trans = Turing.Inference.Transition(vis, lp_proposal, true)
    # state = init_AMGSState(trans, theta, Σ)

    return new_state.transition, new_state 
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_Sampler}, state::newAMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `state::newAMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:new_AMGS_Sampler},
    state::newAMGSState;
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
        # new_state = AMGSState(state.transition, state.varinfo, state.μ, Σ, state.step, state.β, lp_old)
        return state.transition, state
    end

    # Create a new transition and state object if accepted
    # trans = Turing.Inference.Transition(vis, lp_proposal, true)
    # new_state = AMGSState(trans, state.μ, Σ, state.step, state.β, lp_proposal)

    return proposal_state.transition, proposal_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:new_AMGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:new_AMGS_Sampler};
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
    end    # Link the variables

    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)

    # Create a transition object in case of rejection
    prev_transition = Turing.Inference.Transition(model, vis, nothing)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    if(!spl.alg.use_custom_init)
        Σ = 0.01 .* diagm(abs.(theta)) + (1e-6 * I)
    else
        # If we are using a custom initialisation, we need to use the initial covariance matrix
        Σ = spl.alg.init_covariance
    end

    # Propose a new set of parameters and grab the log density 
    orig_state = init_newAMGSState(prev_transition, vis, theta, Σ) 
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

    # Construct the initial state of the sampler
    # new_state = init_AMGSState(new_trans, theta, Σ)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, -(lp_proposal - lp_old)))
    _, upd_state = update_kernel(accep ? proposal_state : orig_state, new_trans, spl, prob)
    
    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:new_AMGS_Sampler}, state::newAMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGS sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
- `state::newAMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:new_AMGS_Sampler},
    state::newAMGSState;
    kwargs...
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Propose the new parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, exp(state.β) .* state.kernel)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : state.transition 

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    _, new_state = update_kernel(accep ? proposal_state : state, new_trans, spl, prob)
    return new_trans, new_state
end


# Define assume functions for the AMGS sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::DynamicPPL.Sampler{<:new_AMGS_Sampler}, dist::Distribution, vn::DynamicPPL.VarName, vi
)
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::DynamicPPL.Sampler{<:new_AMGS_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:Turing.Inference.Transition}, model::AbstractMCMC.AbstractModel, spl::DynamicPPL.Sampler{<:new_AMGS_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGS sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:Turing.Inference.Transition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:new_AMGS_Sampler}`: The AMGS sampler.
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
    spl::DynamicPPL.Sampler{<:new_AMGS_Sampler},
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
