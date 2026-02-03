# write functions step and step_warmup which are used in the Bayes_AMGS_sampler.
# Write code based on algorithm in https://onlinelibrary.wiley.com/doi/10.1111/anzs.12344

# Start by defining the sampler

struct Bayes_AMGS_Sampler{T_accep <: Real, T_ν <: Real, T_λ <: Real, T_noise_scaling <: Real} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    ν₀::T_ν
    λ_min::T_λ
    f::Function
    use_noise_stabiliser::Bool
    noise_scaling::T_noise_scaling
end

"""
    Bayes_AMGS_Sampler(acceptance_rate::T1, ν₀::T2, λ_min::T3, f, space::Tuple) where {T1 <: Real, T2 <: Real, T3 <: Real}

Create a new Bayes AMGS sampler with the specified acceptance rate, ν₀, λ_min, function f, and space.

# Arguments
- `acceptance_rate::T1`: The target acceptance rate for the sampler.
- `ν₀::T2`: The initial value for ν.
- `λ_min::T3`: The minimum value for λ.
- `f`: A function that takes and returns one integer used in the algorithm to determine the earliest step that information about the kernel is provided (should grow slower than the step, i.e. step - f(step) -> ∞ as step -> ∞).
- `noise_stabiliser::Bool`: (Optional) Whether to use a noise stabiliser, defaults to false.
- `noise_stabiliser_scaling::Real`: (Optional) The scaling of the noise stabiliser, defaults to 1e-6.
"""
function Bayes_AMGS_Sampler(
    acceptance_rate::T1,
    ν₀::T2,
    λ_min::T3,
    f;
) where {T1 <: Real, T2 <: Real, T3 <: Real}
    return Bayes_AMGS_Sampler{T1, T2, T3, Float64}(acceptance_rate, ν₀, λ_min, f, false, 1e-6)
end

# DynamicPPL.default_chain_type(sampler::Bayes_AMGS_Sampler) = MCMCChains.Chains

# Define the state of the sampler
# Contains Important quantities for computation of the updated kernel
"""
    BayesAMGSState

State container used by the Bayes AMGS sampler to store running statistics
for Bayesian adaptive MCMC with complex adaptation schemes.

# Fields
- `transition`: the latest `DynamicPPL.ParamsWithStats` object.
- `varinfo`: the `VarInfo` corresponding to the current parameter values.
- `μ`: running mean vector of parameters.
- `kernel`: running covariance (or scale) matrix used for proposals.
- `step`: integer step counter for adaptation scheduling.
- `λ`: current adaptation scaling parameter.
- `λ_start`: initial adaptation scaling parameter.
- `n_start`: starting step for adaptation.
- `A`: adaptation matrix or statistic.
- `δ`: adaptation decay parameter.
- `d`: dimension or degree parameter.
- `c`: adaptation constant.
- `vec_X`: linked list storing historical parameter values.
- `logjoint_internal`: cached log joint density for the current state.
"""
struct BayesAMGSState{
    TTrans<:DynamicPPL.ParamsWithStats,
    TVI<:AbstractVarInfo,
    Tμ_el <: Real,
    Tμ <: AbstractVector{Tμ_el},
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tλ<:Real,
    Tn_start<:Real,
    TA<:Real,
    Tδ<:Real,
    Td<:Int,
    Tc<:Real,
    Tμ_like_el <: Real,
    Tμ_like<:AbstractVector{Tμ_like_el},
    TVecX<:LinkedList{Tμ_like},
    TL<:Real
}
    transition::TTrans
    varinfo::TVI
    μ::Tμ
    kernel::TKernel
    step::TStep
    λ::Tλ
    λ_start::Tλ
    n_start::Tn_start
    A::TA
    δ::Tδ
    d::Td
    c::Tc
    vec_X::TVecX
    logjoint_internal::TL
end

# Function to index a linked list (equivalent to iterating through)
"""
    get_index_ll(idx::Int, vec_X::LinkedList{T}) where {T}

Get the element at index `idx` from a LinkedList `vec_X`.

# Arguments
- `idx::Int`: The 1-based index to retrieve (must be ≥ 1).
- `vec_X::LinkedList{T}`: The linked list to index.

# Returns
- The element at the specified index.
"""
function get_index_ll(idx::Int, vec_X::LinkedList{T}) where {T}
    if idx < 1
        # @error "Index must be greater than 0"
        throw(ArgumentError("Index must be greater than 0"))
    elseif idx == 1
        return vec_X.head
    end
    
    tmp = vec_X
    for i in 1:idx-1
        _, tmp = iterate(tmp)
    end

    if(typeof(tmp) === Nil{T} || typeof(tmp) === Nil{any})
        # @error "Index out of bounds"
        throw(ArgumentError("Index out of bounds"))
    end

    return tmp.head
end

# Define the initial state of the sampler, use the paper to understand A, δ, d and c
"""
    init_BayesAMGSState(transition::T_trans, μ::T_μ, kernel::T_kernel, n_start::T_nstart, A::T_A, δ::T_δ, d::T_d, c::T_c) where {T_trans <: DynamicPPL.ParamsWithStats, T_μ <: AbstractVector, T_kernel <: AbstractMatrix, T_nstart <: Real, T_A <: Real, T_δ <: Real, T_d <: Int, T_c <: Real}

Initialize the state of the Bayes AMGS sampler with the given transition, mean vector μ, kernel matrix, and other parameters.

# Arguments
- `transition::T_trans`: The initial transition object.
- `μ::T_μ`: The initial mean vector.
- `kernel::T_kernel`: The initial kernel matrix.
- `n_start::T_nstart`: The initial value for n_start.
- `A::T_A`: The initial value for A.
- `δ::T_δ`: The initial value for δ.
- `d::T_d`: The initial value for d.
- `c::T_c`: The initial value for c.
"""
function init_BayesAMGSState(
    transition::T_trans,
    varinfo::TVI,
    μ::T_μ,
    kernel::T_kernel,
    n_start::T_nstart,
    A::T_A,
    δ::T_δ,
    d::T_d,
    c::T_c
) where { T_trans <: DynamicPPL.ParamsWithStats, TVI<:AbstractVarInfo, T_μ <: AbstractVector, T_kernel <: AbstractMatrix, T_nstart <: Real, T_A <: Real, T_δ <: Real, T_d <: Int, T_c <: Real}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    λ_start = λ₀ = one(Float64)
    vec_X = cons(μ, nil(typeof(μ)))
    lp = DynamicPPL.getlogjoint_internal(varinfo)
    return BayesAMGSState{T_trans, TVI, eltype(μ), T_μ, T_kernel, Int, Float64, T_nstart, T_A, T_δ, T_d, T_c, eltype(μ), T_μ, Cons{T_μ}, typeof(lp)}(transition, varinfo, μ, kernel, step, λ₀, λ_start, n_start, A, δ, d, c, vec_X, lp)
end

# Write a function which updates the kernel
"""
    update_kernel(rng::TRNG, state::BayesAMGSState, trans::DynamicPPL.ParamsWithStats, spl::Bayes_AMGS_Sampler, prob::T3) where {T3 <: Real, TRNG <: AbstractRNG}

Update the kernel matrix of the Bayes AMGS sampler state based on the new transition and acceptance probability.

# Arguments
- `state::BayesAMGSState`: The current state of the sampler.
- `trans::DynamicPPL.ParamsWithStats`: The new transition object.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `prob::T3`: The acceptance probability.
"""
function update_kernel(rng::TRNG, state::BayesAMGSState, trans::DynamicPPL.ParamsWithStats, spl::Bayes_AMGS_Sampler, prob::T3) where {T3 <: Real, TRNG <: AbstractRNG}
    # Grab useful quantities from the state
    S = state.kernel
    step = state.step
    X_n = state.varinfo[:]
    X_n1 = state.vec_X.head
    # Add to the head of the linked list
    vec_X = cons(X_n, state.vec_X)

    λ = state.λ
    δ = state.δ

    μ = state.μ

    f_step = spl.f(step)
    f_prev_step = spl.f(step - 1)

    n_start = state.n_start
    λ_start = state.λ_start

    ν = spl.ν₀
    d = state.d
    noise_term = spl.use_noise_stabiliser * spl.noise_scaling

    # Perform the update differently depending on whether we are dropping one of the 
    if step == 1
        X̄ = (X_n + X_n1) ./ 2.0
        S = (1.0 / (ν + state.d + 3.0)) * ((X_n1 * X_n1') + (X_n * X_n') - 2.0 * (X̄ * X̄') + (ν + d + 1.0) * S)
        S = S + diagm(noise_term .* rand(rng, size(X_n)[1]))
    elseif f_step == f_prev_step
        X̄ = ((step - f_step) / (step - f_step + 1.0)) * μ + (1.0 / (step - f_step + 1.0)) * X_n
        S = (1.0 / (step - f_step + ν + d + 2.0)) * ((step - f_step + ν + d + 1.0) * S + (X_n * X_n') + (step - f_step) * (μ * μ') - (step - f_step + 1.0) * (X̄ * X̄'))
        S = S + diagm(noise_term .* rand(rng, size(X_n)[1]))

    elseif f_step == f_prev_step + 1
        Xfn1 = get_index_ll(step - f_step + 2, vec_X)
        X̄ = μ + (1.0 / (step - f_step + 1)) * (X_n - Xfn1)
        S = S + (1.0 / (step - f_step + ν + d + 2.0)) * ((X_n * X_n') - (Xfn1 * Xfn1') + (step - f_step + 1.0) * ((μ * μ') - (X̄ * X̄')))
        S = S + diagm(noise_term .* rand(rng, size(X_n)[1]))
    end

    λ = max(spl.λ_min, λ * exp((δ / (n_start + step)) * (prob - spl.acceptance_rate)))
    
    if abs(λ - λ_start) > log(3)
        λ_start = λ
        n_start = (5.0 / (spl.acceptance_rate * (1.0 - spl.acceptance_rate))) - step
    end
    new_state = BayesAMGSState(trans, state.varinfo, X̄, S, step + 1, λ, λ_start, n_start, state.A, state.δ, state.d, state.c, vec_X, state.logjoint_internal)
    return new_state
end

# Function to set the new parameters (for use with gibbs sampler)
Turing.Inference.isgibbscomponent(::Bayes_AMGS_Sampler) = true 

Turing.Inference.init_strategy(::Bayes_AMGS_Sampler) = DynamicPPL.InitFromPrior()

# Get varinfo
function Turing.Inference.get_varinfo(state::T) where {T<:BayesAMGSState}
    # display(state.transition.params)
    # display(typeof(state.transition.params))
    return state.varinfo
end
# function isgibbscomponent(::T) where {T<:AMGS_Sampler} 
#     return true
# end


function Turing.Inference.setparams_varinfo!!(
    model::DynamicPPL.Model,
    sampler::Bayes_AMGS_Sampler,
    state::BayesAMGSState,
    params::AbstractVarInfo,
)
     # Update the vi
    vi = state.varinfo
    vi = DynamicPPL.unflatten(vi, params[:])
    vi  = last(DynamicPPL.evaluate!!(model, vi)) 

    lp = DynamicPPL.getlogjoint_internal(vi)
    # #update the model state
    new_transition = DynamicPPL.ParamsWithStats(vi, model)
    new_state = BayesAMGSState(new_transition, vi, state.μ, state.kernel, state.step, state.λ, state.λ_start, state.n_start, state.A, state.δ, state.d, state.c, state.vec_X, lp)
    
    # Update the VarInfo with the new log density
    return new_state
end

# Write a function which proposes a new set of parameters
"""
    propose_BayesAMGSLL(rng::T_rng, vi::AbstractVarInfo, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler, S::T) where {T_rng <: AbstractRNG, T <: AbstractMatrix}

Propose a new set of parameters for the Bayes AMGS sampler.

# Arguments
- `rng::T_rng`: The random number generator.
- `vi::AbstractVarInfo`: The variable information.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `S::T`: The covariance matrix for the proposal.
"""
function propose_BayesAMGSLL(
    rng::T_rng,
    vi::AbstractVarInfo,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler,
    state::BayesAMGSState,
    S::T
) where {T_rng <: AbstractRNG, T <: AbstractMatrix}
    # Grab the parameters we are performing inference on
    θ = vi[:]

    # Propose a new set of parameters
    proposal = proposewalk(rng, θ, S)
    proposal_vi = DynamicPPL.unflatten(vi, proposal)

    # Call a function to update the varinfo and evaluate the logdensity
    return Turing.Inference.setparams_varinfo!!(model, spl, state, proposal_vi) 
end

# Write a function which initialises the chain where there are warmup steps
"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler; initial_params=nothing, kwargs...)

Perform an initial warmup step for the Bayes AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler;
    initial_params = Turing.Inference.init_strategy(spl),
    discard_sample = false,
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
    prev_transition = DynamicPPL.ParamsWithStats(vis, model)

    # Get the parameters of interest and create the covariance matrix
    theta = vis[:]
    S = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    # Initialize parameters for the Bayes AMGS algorithm
    a = spl.acceptance_rate
    d = length(theta)
    c = 2.38^2 / d
    n_start = 5 / (a * (1-a))
    A = -quantile(Normal(0,1), a * 0.5)
    δ = (1.0 - 1.0/d) * (sqrt(2 * π) * exp((A^2)/2)/(2 * A)) + 1.0 / (d * a * (1-a))

    orig_state = init_BayesAMGSState(prev_transition, vis, theta, S, n_start, A, δ, d, c)

    # Propose a new set of parameters and grab the log density
    proposal_state = propose_BayesAMGSLL(rng, vis, model, spl, orig_state, S)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    return_transition = discard_sample ? nothing : (accep ? proposal_state.transition : prev_transition)
    return return_transition, accep ? proposal_state : orig_state
end

"""
    AbstractMCMC.step_warmup(rng::AbstractRNG, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler, state::BayesAMGSState; initial_params=nothing, kwargs...)

Perform a warmup step for the Bayes AMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `state::BayesAMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler,
    state::BayesAMGSState;
    discard_sample = false,
    kwargs...
)
    # Grab the VarInfo and logdensity of the current parameters
    vis = state.varinfo
    lp_old = state.logjoint_internal

    # Get the kernel
    S = state.kernel

    # Propose the new parameters and get the log density
    proposal_state = propose_BayesAMGSLL(rng, vis, model, spl, state, state.λ * state.c * S)
    lp_proposal = proposal_state.logjoint_internal

    # Check whether to accept the new parameters and return the previous transition if not
    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    #if !accep
        #new_state = BayesAMGSState(prev_transition, state.μ, state.kernel, state.step, state.λ, state.λ_start, state.n_start, state.A, state.δ, state.d, state.c, state.vec_X)
    #    return prev_transition, state
    #end

    # Create a new transition and state object if accepted
    #trans = DynamicPPL.ParamsWithStats(vis, lp_proposal, true)
    #new_state = BayesAMGSState(trans, state.μ, state.kernel, state.step, state.λ, state.λ_start, state.n_start, state.A, state.δ, state.d, state.c, state.vec_X)
    return_transition = discard_sample ? nothing : (accep ? proposal_state.transition : state.transition)
    return return_transition, accep ? proposal_state : state 
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler; initial_params=nothing, kwargs...)

Perform an initial sampling step for the Bayes AMGS sampler.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `initial_params`: The initial parameters (optional).
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler;
    initial_params = Turing.Inference.init_strategy(spl),
    discard_sample = false,
    kwargs...,
)
    vis = DynamicPPL.VarInfo(model)
    _, vis = DynamicPPL.init!!(rng, model, vis, initial_params)
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

    theta = vis[:]
    S = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    d = length(theta)
    c = 2.38^2 / d

    a = spl.acceptance_rate
    n_start = 5.0 / (a * (1.0-a))
    A = -quantile(Normal(0,1), a * 0.5)
    δ = (1.0 - (1.0 / d)) * (sqrt(2 * π) * exp((A^2.0) / 2.0) / (2.0 * A)) + 1.0 / (d * a * (1.0 - a))

    # Create the initial transition object
    prev_transition = DynamicPPL.ParamsWithStats(vis, model, nothing)
    orig_state = init_BayesAMGSState(prev_transition, vis, theta, S, n_start, A, δ, d, c)

    proposal_state = propose_BayesAMGSLL(rng, vis, model, spl, orig_state, c * S)
    lp_proposal = proposal_state.logjoint_internal

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    new_trans = accep ? proposal_state : prev_transition
    prob = exp(min(0.0,-(lp_proposal - lp_old)))
    upd_state = update_kernel(rng, accep ? proposal_state : orig_state, new_trans, spl, prob)
    return_transition = discard_sample ? nothing : new_trans
    return return_transition, upd_state
end

"""
    AbstractMCMC.step(rng::AbstractRNG, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler, state::BayesAMGSState; initial_params=nothing, kwargs...)

Perform a sampling step for the Bayes AMGS sampler for a given state.

# Arguments
- `rng::AbstractRNG`: The random number generator.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `state::BayesAMGSState`: The current state of the sampler.
- `kwargs...`: Additional keyword arguments.
"""
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler,
    state::BayesAMGSState;
    discard_sample = false,
    kwargs...,
)
    vis = state.varinfo
    lp_old = state.logjoint_internal

    proposal_state = propose_BayesAMGSLL(rng, vis, model, spl, state, state.λ * state.c * state.kernel)
    lp_proposal = proposal_state.logjoint_internal

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    new_trans = accep ? proposal_state.transition : state.transition

    prob = exp(min(0.0,(lp_proposal - lp_old)))
    new_state = update_kernel(rng, state, new_trans, spl, prob)
    return_transition = discard_sample ? nothing : new_trans
    return return_transition, new_state
end

#function DynamicPPL.assume(
#    rng::Random.AbstractRNG, spl::Bayes_AMGS_Sampler, dist::Distribution, vn::DynamicPPL.VarName, vi
#)
#    # DynamicPPL.updategid!(vi, vn, spl)
#    return DynamicPPL.assume(dist, vn, vi)
#end
 
getlogevidence(transitions, sampler::Bayes_AMGS_Sampler, state) = missing

"""
    AbstractMCMC.bundle_samples(ts::Vector{<:DynamicPPL.ParamsWithStats}, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler, state, chain_type::Type{MCMCChains.Chains}; save_state=false, stats=missing, sort_chain=false, include_varname_to_symbol=true, discard_initial=0, thinning=1, kwargs...)

Bundle the samples from the Bayes AMGS sampler into a chain.

# Arguments
- `ts::Vector{<:DynamicPPL.ParamsWithStats}`: The Vector of transitions.
- `model::DynamicPPL.Model`: The model.
- `spl::Bayes_AMGS_Sampler`: The Bayes AMGS sampler.
- `state`: The current state of the sampler.
- `chain_type::Type{MCMCChains.Chains}`: The chain type.
- `save_state`: Whether to save the state.
- `stats`: The statistics.
- `sort_chain`: Whether to sort the chain.
- `include_varname_to_symbol`: Whether to include the variable name to symbol mapping.
- `discard_initial`: The number of initial samples to discard.
- `thinning`: The thinning interval.
- `kwargs...`: Additional keyword arguments.
"""
# function AbstractMCMC.bundle_samples(
#     ts::Vector{<:DynamicPPL.ParamsWithStats},
#     model::DynamicPPL.Model,
#     spl::Bayes_AMGS_Sampler,
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
