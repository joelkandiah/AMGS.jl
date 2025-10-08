#  write a functions step and step_warmup which are used in the AMGS_DA_sampler.
# Based on Algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y

# Start by defining the sampler

struct AMGS_DA_Sampler{ T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_B, T_μ <: Real, T_γ <: Real, T_t <: Real, T_κ <: Real} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    μ::T_μ
    γ::T_γ
    t0::T_t
    κ::T_κ
end

function AMGS_DA_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
) where {T1 <: Real, T2 <: Real}
return AMGS_DA_Sampler(acceptance_rate, convergence_step, Val(false), 1e-6, 0.0, 0.05, 10.0, 0.75)
end

function AMGS_DA_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real }
return AMGS_DA_Sampler(acceptance_rate, convergence_step, Val(use_noise_stabiliser), noise_scaling, 0.0, 0.05, 10.0, 0.75)
end


DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_DA_Sampler}) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
struct AMGSDAState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    Tμ<:AbstractVector,
    TKernel<:AbstractMatrix,
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
    Δ_cache::Tμ
    x̄::Tβ
    ḡ::Tβ
end

# Define the initial state of the sampler
"""
    init_AMGSDAState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: AbstractMatrix}

Initialize the state of the AMGSDA sampler with the given transition, mean vector μ, and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_AMGSDAState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    AMGSDAState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, μ, kernel, step, β, lp, similar(μ), 0, 0)
end

"""
    init_AMGSDAState(transition::T_trans, μ::T_μ, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: AbstractMatrix}

Initialize the state of the AMGSDA sampler with the given transition, mean value μ, and kernel matrix for a 1D case.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean value.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_AMGSDAState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel
) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    new_μ = [μ]
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    AMGSDAState{T_trans, TVI, typeof(new_μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, new_μ, kernel, step, β, lp, similar(μ), 0, 0)
end

"""
    update_kernel(state::AMGSDAState, trans::T, spl::Sampler{<:AMGS_DA_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real}

Update the kernel matrix, and internal quantities of the AMGSDA sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSDAState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(state::AMGSDAState, trans::T, spl::Sampler{<:AMGS_DA_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real}
    μ    = state.μ
    step = state.step
    state.Δ_cache .= state.varinfo[:]
    n    = length(state.Δ_cache)

    # Robbins–Monro gain
    @fastmath begin
        η = (step + 1.0)^(-spl.alg.convergence_step)

        # Adapt acceptance-rate scaling
        # β = state.β + η * (prob - spl.alg.acceptance_rate)

        # Update running mean μ ← (1−η)μ + ηθ
        @. μ += η * (state.Δ_cache - μ)

        # Δ = θ - μ (store in preallocated buffer Δ in the state)
        @. state.Δ_cache -= μ

        # Outer product update, symmetric and in-place:
        @inbounds for i in 1:n
            Δi = state.Δ_cache[i]
            for j in i:n
                #ΔΔ = Δ[i] * Δ[j]
                #Σ_ij = state.kernel[i,j]
                state.kernel[i,j] += η * (Δi * state.Δ_cache[j] - state.kernel[i,j])
                state.kernel[j,i] = state.kernel[i,j]  # preserve symmetry
            end
        end

        # Optional noise stabilisation (fast Val dispatch)
        @inline _add_noise!(state.kernel, spl.alg.noise_scaling, n, spl.alg.use_noise_stabiliser)

        ## Dual averaging step
        Ht = spl.alg.acceptance_rate - prob
        ḡ = (1 - 1/(step + spl.alg.t0)) * state.ḡ + (1/(step + spl.alg.t0)) * Ht
        x = spl.alg.μ - ((sqrt(step) / spl.alg.γ) * ḡ)
        x̄ = step^(-spl.alg.κ) * x + (1 - step^(-spl.alg.κ)) * state.x̄
    end
    return state.kernel, AMGSDAState(trans, state.varinfo, μ, state.kernel, step + 1, x, state.logjoint_internal, state.Δ_cache, x̄, ḡ)
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::AMGS_DA_Sampler) = true 

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:AMGSDAState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:AMGS_DA_Sampler},
    state::AMGSDAState,
    params::AbstractVarInfo,
)

    # Update the vi
    # vi = state.varinfo
    # vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, params)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = Turing.Inference.Transition(model, vi, nothing)
    new_state = AMGSDAState(new_transition, vi, state.μ, state.kernel, state.step, state.β, lp, state.Δ_cache, state.x̄, state.ḡ) 
    
    # Update the VarInfo with the new log density
    return new_state
end

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_DA_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGSDA sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `state::AMGSDAState`: The current state of the sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_DA_Sampler},
    state::AMGSDAState,
    Σ::AbstractMatrix
)
    # Grab the parameters we are performing inference on
    state.Δ_cache .= vi[:]
    # Propose a new set of parameters
    proposal = proposewalk(rng, state.Δ_cache, Σ)
    
    proposal_vi = DynamicPPL.unflatten(vi, proposal)

    # Call a function to update the varinfo and evaluate the logdensity
    return Turing.Inference.setparams_varinfo!!(model, spl, state, proposal_vi) 
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_DA_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGSDA sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_DA_Sampler};
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
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    # get the old state:
    orig_state = init_AMGSDAState(prev_transition, vis, theta, Σ)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    return accep ? new_state.transition : prev_transition, accep ? new_state : orig_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_DA_Sampler}, state::AMGSDAState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGSDA sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `state::AMGSDAState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_DA_Sampler},
    state::AMGSDAState;
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

    return accep ? proposal_state.transition : state.transition, accep ? proposal_state : state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_DA_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGSDA sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_DA_Sampler};
    initial_params=nothing,
    kwargs...,
)
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
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    orig_state = init_AMGSDAState(prev_transition, vis, theta, Σ) 
    # Propose a new set of parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : prev_transition

    # Construct the initial state of the sampler
    # new_state = init_AMGSDAState(new_trans, theta, Σ)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    _, upd_state = update_kernel(accep ? proposal_state : orig_state, new_trans, spl, prob)
    
    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_DA_Sampler}, state::AMGSDAState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGSDA sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
- `state::AMGSDAState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_DA_Sampler},
    state::AMGSDAState;
    kwargs...
)
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


# Define assume functions for the AMGSDA sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::DynamicPPL.Sampler{<:AMGS_DA_Sampler}, dist::Distribution, vn::DynamicPPL.VarName, vi
)
    # DynamicPPL.updategid!(vi, vn, spl)
    return DynamicPPL.assume(dist, vn, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::DynamicPPL.Sampler{<:AMGS_DA_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:Turing.Inference.Transition}, model::AbstractMCMC.AbstractModel, spl::DynamicPPL.Sampler{<:AMGS_DA_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGSDA sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:Turing.Inference.Transition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:AMGS_DA_Sampler}`: The AMGSDA sampler.
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
    spl::DynamicPPL.Sampler{<:AMGS_DA_Sampler},
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

# # Handle setting `nadapts` and `discard_initial`
# function AbstractMCMC.sample(
#     rng::AbstractRNG,
#     model::DynamicPPL.Model,
#     sampler::Sampler{<:AMGS_DA_Sampler},
#     N::Integer;
#     chain_type=DynamicPPL.default_chain_type(sampler),
#     resume_from=nothing,
#     initial_state=DynamicPPL.loadstate(resume_from),
#     progress=PROGRESS[],
#     num_warmup = num_warmup,
#     discard_initial = discard_initial,
#     kwargs...,
# )
#     if resume_from === nothing
# 
#         # If `discard_initial` is `-1`, then users did not specify the keyword argument.
# 
#         return AbstractMCMC.mcmcsample(
#             rng,
#             model,
#             sampler,
#             N;
#             chain_type=chain_type,
#             progress=progress,
#             num_warmup = num_warmup,
#             discard_initial = discard_initial,
#             kwargs...,
#         )
#     else
#         return AbstractMCMC.mcmcsample(
#             rng,
#             model,
#             sampler,
#             N;
#             chain_type=chain_type,
#             initial_state=initial_state,
#             progress=progress,
#             num_warmup = 0,
#             discard_initial=0,
#             kwargs...,
#         )
#     end
# end
