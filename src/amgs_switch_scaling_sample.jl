#  write a functions step and step_warmup which are used in the AMGS_SGS_sampler.
# Based on Algorithm 4 in https://link.springer.com/article/10.1007/s11222-008-9110-y
# But we allow the global scaling algorithm to be set using different schemes.
# In particular we are interested in the dual averaging scheme from Hoffman and Gelman (2014)
# https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf
# The idea is to use some method of estimating the scale parameter λ s.t. E[Α(lambda) -α^*] = 0 where α^* is the target acceptance rate and \Alpha is some random variable depending on λ.

# Start by defining the sampler

"""
    AMGS_SGS_Sampler(acceptance_rate, convergence_step; use_noise_stabiliser=false, noise_scaling=1e-6, μ=0.0, γ=0.05, t0=10.0, κ=0.75)

Data Augmentation variant of the Adaptive Metropolis-within-Gibbs sampler.

This sampler implements an adaptive MCMC algorithm designed for data augmentation
contexts, using additional adaptation parameters for improved performance in
hierarchical models or models with latent variables.

# Fields
- `acceptance_rate::Real`: target acceptance rate for the Robbins–Monro update.
- `convergence_step::Real`: controls the step-size decay in adaptation (power).
- `use_noise_stabiliser::Val{Bool}`: fast-dispatch flag to optionally add noise.
- `noise_scaling::Real`: variance scale used by the optional noise stabiliser.
- `μ::Real`: adaptation parameter for mean updates.
- `γ::Real`: adaptation parameter for covariance scaling.
- `t0::Real`: initial adaptation time parameter.
- `κ::Real`: adaptation decay parameter.

# Examples
```julia
sampler = AMGS_SGS_Sampler(0.234, 1000)
chain = sample(model, sampler, 1000)
```

TODO: Add detailed explanation of data augmentation context and parameter meanings.
TODO: Explain the adaptation scheme differences from standard AMGS.
"""
struct AMGS_SGS_Sampler{ T_accep <: Real, T_conv_step <: Real, T_scale <: Real, T_B, T_μ <: Real, T_γ <: Real, T_t <: Real, T_κ <: Real, M} <: Turing.InferenceAlgorithm
    acceptance_rate::T_accep
    convergence_step::T_conv_step
    use_noise_stabiliser::Val{T_B}
    noise_scaling::T_scale
    μ::T_μ
    γ::T_γ
    t0::T_t
    κ::T_κ
    method::Val{M}
end

function AMGS_SGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2;
    method::Symbol = :DA
) where {T1 <: Real, T2 <: Real}
return AMGS_SGS_Sampler(acceptance_rate, convergence_step, Val(false), 1e-6, 0.0, 0.05, 10.0, 0.75, Val(method))
end

function AMGS_SGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    use_noise_stabiliser::Bool,
    noise_scaling::T3;
    method::Symbol = :DA
) where {T1 <: Real, T2 <: Real, T3 <: Real }
return AMGS_SGS_Sampler(acceptance_rate, convergence_step, Val(use_noise_stabiliser), noise_scaling, 0.0, 0.05, 10.0, 0.75, Val(method))
end


DynamicPPL.default_chain_type(sampler::Sampler{<:AMGS_SGS_Sampler}) = MCMCChains.Chains

# Define the state of the global scale adaptation
abstract type GlobalScalingAdaptationState{T<:Real} end

struct RobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

struct DualAveragingState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
    ḡ::T
end

struct ExponentialMovingAverageState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

struct LogisticRobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

struct NewtonRaphsonRobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
end

struct ScaleInvariantNewtonRaphsonRobbinsMonroState{T<:Real} <: GlobalScalingAdaptationState{T}
    β::T
    v̂::T
end

# Type-parametric initializers
init_global_adaptation_state(::Val{:RM}, ::Type{T}) where {T<:Real} = RobbinsMonroState{T}(zero(T))
init_global_adaptation_state(::Val{:DA}, ::Type{T}) where {T<:Real} = DualAveragingState{T}(zero(T), zero(T))
init_global_adaptation_state(::Val{:EMA}, ::Type{T}) where {T<:Real} = ExponentialMovingAverageState{T}(zero(T))
init_global_adaptation_state(::Val{:LOG}, ::Type{T}) where {T<:Real} = LogisticRobbinsMonroState{T}(zero(T))
init_global_adaptation_state(::Val{:NR}, ::Type{T}) where {T<:Real} = NewtonRaphsonRobbinsMonroState{T}(zero(T))
init_global_adaptation_state(::Val{:SINR}, ::Type{T}) where {T<:Real} = ScaleInvariantNewtonRaphsonRobbinsMonroState{T}(zero(T), one(T))
init_global_adaptation_state(::Val{:CSINR}, ::Type{T}) where {T<:Real} = ScaleInvariantNewtonRaphsonRobbinsMonroState{T}(zero(T), one(T))

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
"""
    AMGSSGSState

State container used by the AMGS Data Augmentation sampler to store running statistics
required for adaptation in data augmentation contexts.

# Fields
- `transition`: the latest `Turing.Inference.Transition` object.
- `varinfo`: the `VarInfo` corresponding to the current parameter values.
- `μ`: running mean vector of parameters.
- `kernel`: running covariance (or scale) matrix used for proposals.
- `step`: integer step counter for adaptation scheduling.
- `logjoint_internal`: cached log joint density for the current state.
- `Δ_cache`: preallocated buffer used for delta computations.
"""
struct AMGSSGSState{
    TTrans<:Turing.Inference.Transition,
    TVI<:AbstractVarInfo,
    Tμ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    TL<:Real,
    T_GSAS <: GlobalScalingAdaptationState
}
    transition::TTrans
    varinfo::TVI
    μ::Tμ
    kernel::TKernel
    step::TStep
    logjoint_internal::TL
    Δ_cache::Tμ
    global_scale_adaptation_state::T_GSAS
end

# Define the initial state of the sampler
"""
    init_AMGSSGSState(transition::T_trans, μ::Vector{T_μ}, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: AbstractMatrix}

Initialize the state of the AMGSSGS sampler with the given transition, mean vector μ, and kernel matrix.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::Vector{T_μ}`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_AMGSSGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::Vector{T_μ},
    kernel::T_kernel,
    spl::Sampler{<:AMGS_SGS_Sampler}
    ) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix} 
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    method_state = init_global_adaptation_state(spl.alg.method, eltype(μ))
    AMGSSGSState{T_trans, TVI, typeof(μ), T_kernel, Int, typeof(lp), typeof(method_state)}(transition, varinfo, μ, kernel, step,  lp, similar(μ), method_state)
end

"""
    init_AMGSSGSState(transition::T_trans, μ::T_μ, kernel::T_kernel) where {T_trans <: Turing.Inference.Transition, T_μ <: Real, T_kernel <: AbstractMatrix}

Initialize the state of the AMGSSGS sampler with the given transition, mean value μ, and kernel matrix for a 1D case.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean value.
- `kernel::T_kernel`: The initial kernel matrix.
"""
function init_AMGSSGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel,
    spl::Sampler{<:AMGS_SGS_Sampler}
    ) where {T_trans <: Turing.Inference.Transition, TVI <: AbstractVarInfo, T_μ <: Real, T_kernel <: AbstractMatrix} 
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    new_μ = [μ]
    lp = DynamicPPL.getlogjoint_internal(varinfo) # Get the log joint density
    method_state = init_global_adaptation_state(spl.alg.method, eltype(μ))
    AMGSSGSState{T_trans, TVI, typeof(new_μ), T_kernel, Int, typeof(lp), typeof(method_state)}(transition, varinfo, new_μ, kernel, step, lp, similar(μ), method_state)
end

"""
    update_kernel(rng::TRNG, state::AMGSSGSState, trans::T, spl::Sampler{<:AMGS_SGS_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real, TRNG <: AbstractRNG}

Update the kernel matrix, and internal quantities of the AMGSSGS sampler state based on the new transition and acceptance probability.

# Arguments
- `state::AMGSSGSState`: The current state of the sampler.
- `trans::T`: The new transition object.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(rng::TRNG, state::AMGSSGSState, trans::T, spl::Sampler{<:AMGS_SGS_Sampler}, prob::T3) where {T <: Turing.Inference.Transition, T3 <: Real, TRNG <: AbstractRNG}
    μ    = state.μ
    step = state.step
    state.Δ_cache .= state.varinfo[:]
    n    = length(state.Δ_cache)

    # Robbins–Monro gain
    @fastmath begin
        η = (step + 1.0)^(-spl.alg.convergence_step)

        # Update running mean μ ← (1−η)μ + ηθ
        @. μ += η * (state.Δ_cache - μ)

        # Δ = θ - μ (store in preallocated buffer Δ in the state)
        @. state.Δ_cache -= μ

        # Outer product update, symmetric and in-place:
        # Optimized: Use BLAS symmetric rank-1 update for outer product
        # First scale kernel by (1-η)
        state.kernel .*= (1.0 - η)
        # Then add η*Δ*Δᵀ using symmetric rank-1 update
        BLAS.syr!('U', η, state.Δ_cache, state.kernel)
        # Copy upper triangle to lower triangle for symmetry
        @inbounds for j in 1:n
            for i in (j+1):n
                state.kernel[i,j] = state.kernel[j,i]
            end
        end

        # Optional noise stabilisation (fast Val dispatch)
        @inline _add_noise!(rng, state.kernel, spl.alg.noise_scaling, n, spl.alg.use_noise_stabiliser)

    end

    method_state = adapt_scale(spl.alg.method, spl, state.global_scale_adaptation_state, step, prob)
    return state.kernel, AMGSSGSState(trans, state.varinfo, μ, state.kernel, step + 1, state.logjoint_internal, state.Δ_cache, method_state)
end

function adapt_scale(::Val{:RM}, spl, method_state, step, prob)
        γt = (step + 1.0)^(-spl.alg.convergence_step)
        newβ = method_state.β + γt * (prob - spl.alg.acceptance_rate)
        return RobbinsMonroState{typeof(newβ)}(newβ)
end

# Hoffman Gelman dual averaging scheme
# https://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf
function adapt_scale(::Val{:DA}, spl, method_state, step, prob)
        ## Dual averaging step
        Ht = spl.alg.acceptance_rate - prob
        ḡ = (1 - 1/(step + spl.alg.t0)) * method_state.ḡ + (1/(step + spl.alg.t0)) * Ht
        x = spl.alg.μ - ((sqrt(step) / spl.alg.γ) * ḡ)
        x̄ = step^(-spl.alg.κ) * x + (1 - step^(-spl.alg.κ)) * method_state.β
        return DualAveragingState{typeof(x̄)}(x̄, ḡ)
end
   
# Currently equivalent to RM
function adapt_scale(::Val{:EMA}, spl, method_state, step, prob)
    η = (step + 1.0)^(-spl.alg.convergence_step)
    target = method_state.β + (prob - spl.alg.acceptance_rate)
    newβ = (1 - η) * method_state.β + η * target
    return ExponentialMovingAverageState{typeof(newβ)}(newβ)
end
    
function adapt_scale(::Val{:LOG}, spl, method_state, step, prob)
    γt = (step + 1.0)^(-spl.alg.convergence_step)
    tmp = method_state.β + γt * (prob - spl.alg.acceptance_rate)
    newβ = log(1 / (1 + exp(-tmp)))
    return LogisticRobbinsMonroState{typeof(newβ)}(newβ)
end

function adapt_scale(::Val{:NR}, spl, method_state, step, prob)
        γt = (step + 1.0)^(-spl.alg.convergence_step)
        σ2 = clamp(abs(prob * (1 - prob)), 1e-4, 0.25)
        newβ = method_state.β + γt * ((prob - spl.alg.acceptance_rate) / σ2)
        return NewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ)
end

function adapt_scale(::Val{:SINR}, spl, method_state, step, prob)
        γt = (step + 1.0)^(-spl.alg.convergence_step)
        v̂ = (1 - spl.alg.γ) * method_state.v̂ + spl.alg.γ * (prob - spl.alg.convergence_step)^2
        newβ = method_state.β + γt * ((prob - spl.alg.acceptance_rate) / (sqrt(v̂) + 1e-6))
        return ScaleInvariantNewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ, v̂)
end

function adapt_scale(::Val{:CSINR}, spl, method_state, step, prob)
        γt = (step + 1.0)^(-spl.alg.convergence_step)
        σ2 = clamp(prob * (1 - prob), 1e-4, 0.25)
        v̂ = (1 - spl.alg.γ) * method_state.v̂ + spl.alg.γ * (prob - spl.alg.convergence_step)^2
        denom = max(sqrt(v̂), σ2)
        newβ = method_state.β + γt * ((prob - spl.alg.acceptance_rate) / denom)
        return ScaleInvariantNewtonRaphsonRobbinsMonroState{typeof(newβ)}(newβ, v̂)
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::AMGS_SGS_Sampler) = true 

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:AMGSSGSState}
    return state.varinfo
end

function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Sampler{<:AMGS_SGS_Sampler},
    state::AMGSSGSState,
    params::AbstractVarInfo,
)

    # Update the vi
    vi  = last(DynamicPPL.evaluate!!(model, params)) 

    lp = DynamicPPL.getlogjoint_internal(vi)

    # #update the model state
    new_transition = Turing.Inference.Transition(model, vi, nothing)
    new_state = AMGSSGSState(new_transition, vi, state.μ, state.kernel, state.step, lp, state.Δ_cache, state.global_scale_adaptation_state)
    
    # Update the VarInfo with the new log density
    return new_state
end

# Write a function which proposes a new set of parameters
"""
    propose_amgs(rng::AbstractRNG, vi::AbstractVarInfo, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_SGS_Sampler}, Σ::AbstractMatrix)

Propose a new set of parameters for the AMGSSGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `state::AMGSSGSState`: The current state of the sampler.
- `Σ::AbstractMatrix`: The covariance matrix for the proposal.
"""
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_SGS_Sampler},
    state::AMGSSGSState,
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
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_SGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial warmup step for the AMGSSGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_SGS_Sampler};
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
    orig_state = init_AMGSSGSState(prev_transition, vis, theta, Σ, spl)

    # Propose a new set of parameters and grab the log density
    new_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = new_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    return accep ? new_state.transition : prev_transition, accep ? new_state : orig_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::AbstractMCMC.AbstractModel, spl::Sampler{<:AMGS_SGS_Sampler}, state::AMGSSGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the AMGSSGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `state::AMGSSGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_SGS_Sampler},
    state::AMGSSGSState;
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
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_SGS_Sampler}; initial_params=nothing, kwargs...)

Perform an initial sampling step for the AMGSSGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_SGS_Sampler};
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

    orig_state = init_AMGSSGSState(prev_transition, vis, theta, Σ, spl) 
    # Propose a new set of parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, orig_state, Σ)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters 
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    # Construct the transition object to be returned
    new_trans = accep ? proposal_state.transition : prev_transition

    # Update the kernel and return the new transition and state
    prob = exp(min(0.0, (lp_proposal - lp_old)))
    _, upd_state = update_kernel(rng, accep ? proposal_state : orig_state, new_trans, spl, prob)
    
    return new_trans, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Sampler{<:AMGS_SGS_Sampler}, state::AMGSSGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the AMGSSGS sampler given a state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
- `state::AMGSSGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_SGS_Sampler},
    state::AMGSSGSState;
    kwargs...
)
    vis = state.varinfo
    lp_old = state.logjoint_internal
    # Propose the new parameters and grab the log density
    proposal_state = propose_amgs(rng, vis, model, spl, state, exp(state.global_scale_adaptation_state.β) .* state.kernel)
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


# Define assume functions for the AMGSSGS sampler
function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::DynamicPPL.Sampler{<:AMGS_SGS_Sampler}, dist::Distribution, vn::DynamicPPL.VarName, vi
)
    return DynamicPPL.assume(dist, vn, vi)
end

# Define function to get log evidence
getlogevidence(transitions, sampler::DynamicPPL.Sampler{<:AMGS_SGS_Sampler}, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:Turing.Inference.Transition}, model::AbstractMCMC.AbstractModel, spl::DynamicPPL.Sampler{<:AMGS_SGS_Sampler}, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the AMGSSGS sampler into an MCMCChains Chain.

# Arguments
- `ts::Vector{<:Turing.Inference.Transition}`: The Vector of transitions.
- `model::AbstractMCMC.AbstractModel`: The model.
- `spl::DynamicPPL.Sampler{<:AMGS_SGS_Sampler}`: The AMGSSGS sampler.
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
    spl::DynamicPPL.Sampler{<:AMGS_SGS_Sampler},
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

