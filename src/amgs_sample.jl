# write a functions step and step_warmup which are used in the AMGS_sampler.

# Start by defining the sampler

struct AMGS_Sampler{space, T_accep <: Real, T_conv_step <: Real} <: AbstractMCMC.AbstractSampler
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

function AMGS_Sampler(
    acceptance_rate::T1,
    convergence_step::T2,
    space::Symbol...;
)  where {T1 <: Real, T2 <: Real}
    return AMGS_Sampler(acceptance_rate, convergence_step, space;)
end

DynamicPPL.getspace(::AMGS_Sampler{space, <: Any, <: Any}) where {space} = space

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
function init_AMGSState(
    transition::T_trans,
    μ::Vector{T_μ},
    kernel::T_kernel
) where {T_trans <: AdvancedMH.AbstractTransition, T_μ <: Real, T_kernel <: Matrix}
    ## copy vector length of mu and initialise zero vector for beta
    step = 1
    β =  zero(eltype(μ))
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
    β =  zero(typeof(μ))
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
function update_kernel(state::AMGSState, trans::T, spl::Sampler{<:AMGS_Sampler}, prob::T3) where {T <: AdvancedMH.Transition, T3 <: Real}
    Σ = state.kernel
    μ = state.μ
    step = state.step
    β = state.β
    η = (step+1.0)^(-spl.alg.convergence_step)
    β = β + η * (prob - spl.alg.acceptance_rate)
    Δ = trans.params[spl] - μ
    μ = (1.0-η) * μ + (η * trans.params[spl])
    Σ = (1.0-η) * Σ + (η * (Δ * (Δ')))
    new_state = AMGSState(trans, μ, Σ, step + 1, β)
    return new_state
end

# Define a function which proposes a new set of parameters
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, Σ::AbstractMatrix)
    return rand(rng, MvNormal(θ, Σ))
end

# Write a function which checks whether to accept a new set of parameters
mh_accept_logdens(rng::T, s_prime::T2, s::T2) where {T <:AbstractRNG, T2 <: Real} = -randexp() < min(1e-6, s_prime - s)

# Write a function which proposes a new set of parameters
function propose_amgs(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
    Σ::AbstractMatrix
)
    θ = vi[spl]

    proposal = proposewalk(rng, θ, Σ)

    new_ldf = Base.Fix1(
        Turing.LogDensityProblems.logdensity,
        Turing.LogDensityFunction(
            model,
            vi,
            DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context))
        )
    )

    lp_proposal = new_ldf(proposal)

    vi = DynamicPPL.unflatten(vi,
           DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
           proposal)

    return setlogp!!(vi, lp_proposal)
end

# Write a function which initialises the chain where there are warmup steps
function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler};
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = VarInfo()
    spl_init = DynamicPPL.initialsampler(spl)
    model(rng, vis, spl_init)

    DynamicPPL.link!!(vis, spl, model)

    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    theta = vis[spl]
    Σ = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        return prev_transition, init_AMGSState(prev_transition, theta, Σ)
    end
    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    state = init_AMGSState(trans, theta, Σ)

    return trans, state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
    state::AMGSState;
    kwargs...
)
    vis = state.transition.params
    lp_old = getlogp(vis)

    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    Σ = state.kernel

    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)
    if !accep
        new_state = AMGSState(prev_transition, state.μ, Σ, state.step, state.β)
        return prev_transition, state
    end

    trans = AdvancedMH.Transition(vis, lp_proposal, true)
    new_state = AMGSState(trans, state.μ, Σ, state.step, state.β)

    return trans, new_state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Sampler{<:AMGS_Sampler};
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = VarInfo()
    spl_init = DynamicPPL.initialsampler(spl)
    model(rng, vis, spl_init)

    DynamicPPL.link!!(vis, spl, model)

    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    theta = vis[spl]
    Σ = 0.01 .* abs.(diagm(theta)) + (1e-6 * I)

    vis = propose_amgs(rng, vis, model, spl, Σ)
    lp_proposal = getlogp(vis)

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    if !accep
        new_trans = prev_transition
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end

    new_state = init_AMGSState(new_trans, theta, Σ)

    prob = exp(min(0.0,-(lp_proposal - lp_old)))
    upd_state = update_kernel(new_state, new_trans, spl, prob)

    return new_trans, upd_state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::AbstractMCMC.AbstractModel,
    spl::Sampler{<:AMGS_Sampler},
    state::AMGSState;
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vis = state.transition.params
    lp_old = getlogp(vis)
    prev_transition = AdvancedMH.Transition(vis, lp_old, false)

    theta = vis[spl]

    vis = propose_amgs(rng, vis, model, spl, exp(state.β) * state.kernel)
    lp_proposal = getlogp(vis)

    accep = mh_accept_logdens(rng, lp_proposal, lp_old)

    if !accep
        new_trans = prev_transition
    else
        new_trans = AdvancedMH.Transition(vis, lp_proposal, true)
    end

    prob = exp(min(0.0,(lp_proposal - lp_old)))
    new_state = update_kernel(state, new_trans, spl, prob)

    return new_trans, new_state
end

function DynamicPPL.assume(
    rng::Random.AbstractRNG, spl::Turing.Sampler{<:AMGS_Sampler}, dist::Distribution, vn::Turing.VarName, vi
)
    DynamicPPL.updategid!(vi, vn, spl)
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
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dist, var, vns, vi)
end

function DynamicPPL.dot_assume(
    rng,
    spl::Turing.Sampler{<:AMGS_Sampler},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:Turing.VarName},
    var::AbstractArray,
    vi,
)
    # Just defer to `SampleFromPrior`.
   DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    return DynamicPPL.dot_assume(dists, var, vns, vi)
end

function DynamicPPL.observe(spl::Turing.Sampler{<:AMGS_Sampler}, d::Distribution, value, vi)
    return DynamicPPL.observe(d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Turing.Sampler{<:AMGS_Sampler},
    ds::Union{Distribution,AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(ds, value, vi)
end

getlogevidence(transitions, sampler, state) = missing

function _params_to_array(model::DynamicPPL.Model, ts::Vector)
    names_set = OrderedSet{VarName}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        nms_and_vs = Turing.Inference.getparams(model, t.params)
        nms = map(first, nms_and_vs)
        vs = map(last, nms_and_vs)
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return OrderedDict(zip(nms, vs))
    end
    names = collect(names_set)
    vals = [
        get(dicts[i], key, missing) for i in eachindex(dicts), (j, key) in enumerate(names)
    ]

    return names, vals
end


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
