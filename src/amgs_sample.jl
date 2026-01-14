"""
amgs_sample.jl

Sample implementations for the Adaptive Metropolis-within-Gibbs (AMGS)
sampler and variants.

This file defines the `AMGS_Sampler` type and the core `step` /
`step_warmup` implementations used by AbstractMCMC/DynamicPPL. Many
functions include TODO placeholders where a more detailed algorithmic
description or mathematical derivation should be inserted.

References
- https://link.springer.com/article/10.1007/s11222-008-9110-y 
- [Add more references and explanation here]

"""

# Start by defining the sampler

"""
    AMGS_Sampler(acceptance_rate, convergence_step; use_noise_stabiliser=false, noise_scaling=1e-6)

Sampler type for the adaptive Metropolis-within-Gibbs algorithm.

# Fields
- `acceptance_rate::Real`: target acceptance rate for the Robbins–Monro update.
- `convergence_step::Real`: controls the step-size decay in adaptation (power).
- `use_noise_stabiliser::Val{Bool}`: fast-dispatch flag to optionally add noise.
- `noise_scaling::Real`: variance scale used by the optional noise stabiliser.

TODO: Add an example of constructing the sampler and a brief note on how to
select sensible defaults for `acceptance_rate` and `convergence_step`.
"""
struct AMGS_Sampler{ T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_B} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
end

function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
) where {T1 <: Real, T2 <: Real}
return AMGS_Sampler(acceptance_rate, convergence_step, Val(false), 1e-6)
end

function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3
) where {T1 <: Real, T2 <: Real, T3 <: Real }
return AMGS_Sampler(acceptance_rate, convergence_step, Val(use_noise_stabiliser), noise_scaling)
end


# DynamicPPL.default_chain_type(sampler::AMGS_Sampler) = MCMCChains.Chains

"""
    AMGSState

State container used by the AMGS sampler to store running statistics
required for adaptation.

# Fields
- `transition`: the latest `DynamicPPL.ParamsWithStats` object.
- `varinfo`: the `VarInfo` corresponding to the current parameter values.
- `μ`: running mean vector of parameters.
- `kernel`: running covariance (or scale) matrix used for proposals.
- `step`: integer step counter for adaptation scheduling.
- `β`: scalar adapt scaling for acceptance rate control.
- `logjoint_internal`: cached log joint density for the current state.
- `Δ_cache`: preallocated buffer used for delta computations.
"""
struct AMGSState{
    TTrans<:DynamicPPL.ParamsWithStats,
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
end

# Define the initial state of the sampler
"""
    init_AMGSState(transition, varinfo, μ, kernel)

Create a new `AMGSState` initializing running mean and kernel.

The `μ` vector should have the same length as the number of parameters in the model,
typically initialized to zeros or the starting parameter values. The `kernel` matrix
should be a positive definite covariance matrix, often initialized as a diagonal
matrix with variances reflecting expected parameter scales (e.g., `diagm([1.0, 1.0])`
for a 2-parameter model).

For hierarchical models, `μ` and `kernel` should account for all flattened parameters
after `DynamicPPL.varname_and_value_leaves` processing.
"""
function init_AMGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel
) where {T_trans <: DynamicPPL.ParamsWithStats, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(eltype(μ))
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    AMGSState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, μ, kernel, step, β, lp, similar(μ))
end

"""
    init_AMGSState(transition, varinfo, μ, kernel)  # 1D convenience constructor

Convenience constructor when `μ` is scalar (1D parameter vector).
"""
function init_AMGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel
) where {T_trans <: DynamicPPL.ParamsWithStats, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β = zero(typeof(μ))
    new_μ = [μ]
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    AMGSState{T_trans, TVI, typeof(new_μ), T_kernel, Int, typeof(β), typeof(lp)}(transition, varinfo, new_μ, kernel, step, β, lp, similar(μ))
end

# Write a function which updates the kernel
"""
    _add_noise!(rng, Δ², noise_scaling, θ_size, use_noise_stabiliser)

Add diagonal noise to `Δ²` when `use_noise_stabiliser` is true. This can help
prevent degeneracy of the running covariance matrix during adaptation.

In high-dimensional or poorly conditioned problems, the empirical covariance
matrix may become singular or nearly singular. Adding small diagonal noise
helps maintain numerical stability and prevents the Cholesky decomposition
from failing. The noise is drawn from a distribution scaled by `noise_scaling`.

Recommended `noise_scaling` values are typically small (1e-6 to 1e-4) and should
be much smaller than the typical parameter variances to avoid biasing the
adaptation too strongly.
"""
function _add_noise!(rng::T, Δ², noise_scaling, θ_size, ::Val{true}) where {T <:AbstractRNG}
    Δ² .= Δ² .+ diagm(noise_scaling .* rand(rng, θ_size))
end

function _add_noise!(rng::T, Δ², noise_scaling, θ_size, ::Val{false}) where {T <: AbstractRNG}
    # No-op when stabiliser disabled
end


"""
    update_kernel(rng, state, trans, spl, prob)

Perform a Robbins–Monro style update of the running mean and covariance
(`state.μ` and `state.kernel`) given an acceptance probability `prob`.

This function implements the core adaptation mechanism of AMGS, updating the
proposal distribution online based on the current parameter values and acceptance
probability. The adaptation uses a diminishing step size η = (step+1)^(-γ) where
γ is the `convergence_step` parameter.

The updates follow:
- Mean: μ ← (1-η)μ + ηθ (moving average of parameters)
- Covariance: Σ ← (1-η)Σ + η(θ-μ)(θ-μ)ᵀ (moving average of outer products)
- Acceptance scaling: β ← β + η(prob - target_rate)

Returns the updated kernel and a new `AMGSState`.

TODO: Add mathematical derivation of Robbins-Monro convergence properties.
TODO: Explain the relationship between convergence_step and adaptation speed.
"""
function update_kernel(rng::TRNG, state::AMGSState, trans::T, spl::AMGS_Sampler, prob::T3) where {T <: DynamicPPL.ParamsWithStats, T3 <: Real, TRNG <: AbstractRNG}
    μ    = state.μ
    step = state.step
    state.Δ_cache .= state.varinfo[:]
    n    = length(state.Δ_cache)

    # Robbins–Monro gain
    @fastmath begin
        η = (step + 1.0)^(-spl.convergence_step)

        # Adapt acceptance-rate scaling
        β = state.β + η * (prob - spl.acceptance_rate)

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
        @inline _add_noise!(rng, state.kernel, spl.noise_scaling, n, spl.use_noise_stabiliser)
    end 
    return state.kernel, AMGSState(trans, state.varinfo, μ, state.kernel, step + 1, β, state.logjoint_internal, state.Δ_cache)
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::AMGS_Sampler) = true 

Turing.Inference.init_strategy(::AMGS_Sampler) = DynamicPPL.InitFromPrior()

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:AMGSState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::AMGS_Sampler,
    state::AMGSState,
    params::AbstractVarInfo,
)

    # Update the vi
    # vi = state.varinfo
    # vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, params)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = DynamicPPL.ParamsWithStats(vi, model)
    new_state = AMGSState(new_transition, vi, state.μ, state.kernel, state.step, state.β, lp, state.Δ_cache) 
    
    # Update the VarInfo with the new log density
    return new_state
end

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::AMGS_Sampler, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::AMGS_Sampler,
    state::AMGSState,
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
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::AMGS_Sampler; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = DynamicPPL.VarInfo(model)
    _, vis = DynamicPPL.init!!(rng, model, vis, initial_params)

    # If there are initial parameters, set them and the logdensity
    # if initial_params !== nothing
    #     vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
    #     vis = last(DynamicPPL.evaluate!!(model, vis))
    # end

    # Link the variables
    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)
    prev_transition = DynamicPPL.ParamsWithStats(vis, model)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    # get the old state:
    orig_state = init_AMGSState(prev_transition, vis, theta, Σ)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    return accep ? new_state.transition : prev_transition, accep ? new_state : orig_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::AMGS_Sampler, state::AMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::AMGS_Sampler,
    state::AMGSState;
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
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::AMGS_Sampler; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    kwargs...,
)
    #vis = DynamicPPL.default_varinfo(rng, model, spl)

    vis = DynamicPPL.VarInfo(model)
    _, vis = DynamicPPL.init!!(rng, model, vis, initial_params)

    # If there are initial parameters, set them and the logdensity
   #  if initial_params !== nothing
   #      vis = DynamicPPL.initialize_parameters!!(vis, initial_params, model)
   #      vis = last(DynamicPPL.evaluate!!(model, vis))
   #  end    # Link the variables

    DynamicPPL.link!!(vis, model)
    DynamicPPL.evaluate!!(model, vis)

    # Get the log density of the current parameters
    lp_old = DynamicPPL.getlogjoint_internal(vis)

    # Create a transition object in case of rejection
    prev_transition = DynamicPPL.ParamsWithStats(vis, model)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    Σ = diagm(0.01 .* abs.(theta) .+ 1e-6)

    orig_state = init_AMGSState(prev_transition, vis, theta, Σ) 
    # Propose a new set of parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : prev_transition

    # Construct the initial state of the sampler
    # new_state = init_AMGSState(new_trans, theta, Σ)

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    _, upd_state = update_kernel(rng, accep ? proposal_state : orig_state, new_trans, spl, prob)
    
    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::AMGS_Sampler, state::AMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGS sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
- `state::AMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::AMGS_Sampler,
    state::AMGSState;
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
    _, new_state = update_kernel(rng, accep ? proposal_state : state, new_trans, spl, prob)
    return new_trans, new_state
end


# Define assume functions for the AMGS sampler
# function DynamicPPL.assume(
#     rng::Random.AbstractRNG, spl::AMGS_Sampler, dist::Distribution, vn::DynamicPPL.VarName, vi
# )
#     # DynamicPPL.updategid!(vi, vn, spl)
#     return DynamicPPL.assume(dist, vn, vi)
# end

# Define function to get log evidence
getlogevidence(transitions, sampler::AMGS_Sampler, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:DynamicPPL.ParamsWithStats}, model::AbstractMCMC.AbstractModel, spl::AMGS_Sampler, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGS sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:DynamicPPL.ParamsWithStats}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::AMGS_Sampler`: The AMGS sampler.
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
# function AbstractMCMC.bundle_samples(
#     ts::Vector{<:DynamicPPL.ParamsWithStats},
#     model::DynamicPPL.Model,
#     spl::AMGS_Sampler,
#     state,
#     chain_type::Type{MCMCChains.Chains};
#     save_state=false,
#     stats=missing,
#     sort_chain=false,
#     include_varname_to_symbol=true,
#     discard_initial=0,
#     thinning=1,
#     kwargs...,
# )
# 
#     # Convert transitions to array format.
#     # Also retrieve the variable names.
#     varnames, vals = _params_to_array(model, ts)
#     varnames_symbol = map(Symbol, varnames)
# 
#     # Get the values of the extra parameters in each transition.
#     extra_params, extra_values = Turing.Inference.get_transition_extras(ts)
# 
#     # Extract names & construct param array.
#     nms = [varnames_symbol; extra_params]
#     parray = hcat(vals, extra_values)
# 
#     # Get the average or final log evidence, if it exists.
#     le = getlogevidence(ts, spl, state)
# 
#     # Set up the info tuple.
#     info = NamedTuple()
# 
#     if include_varname_to_symbol
#         info = merge(info, (varname_to_symbol=OrderedDict(zip(varnames, varnames_symbol)),))
#     end
# 
#     if save_state
#         info = merge(info, (model=model, sampler=spl, samplerstate=state))
#     end
# 
#     # Merge in the timing info, if available
#     if !ismissing(stats)
#         info = merge(info, (start_time=stats.start, stop_time=stats.stop))
#     end
# 
#     # Conretize the array before giving it to MCMCChains.
#     parray = MCMCChains.concretize(parray)
# 
#     # Chain construction.
#     chain = MCMCChains.Chains(
#         parray,
#         nms,
#         (internals=extra_params,);
#         evidence=le,
#         info=info,
#         start=discard_initial + 1,
#         thin=thinning,
#     )
# 
#     return sort_chain ? sort(chain) : chain
# end
