"""
    set_namedtuple!(vi, nt)

Populate a `DynamicPPL.VarInfo` (or ThreadSafeVarInfo) `vi` from a `NamedTuple` of
parameter arrays `nt`.

# Arguments
- `vi::DynamicPPL.VarInfoOrThreadSafeVarInfo`: the VarInfo object to modify in-place.
- `nt::NamedTuple`: mapping from variable names to values (vectors or scalars).

# Notes
- If a named entry corresponds to multiple internal variable names, the values are
  spread across them. If lengths mismatch, an error is thrown.

TODO: Add an example showing typical `nt` shapes for hierarchical models.
"""
function set_namedtuple!(vi::DynamicPPL.VarInfoOrThreadSafeVarInfo, nt::NamedTuple)
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns
        if vals isa AbstractVector
            vals = unvectorize(vals)
        end
        if length(vns) == 1
            # Only one variable, assign the values to it
            DynamicPPL.setindex!(vi, vals, vns[1])
        else
            # Spread the values across the variables
            length(vns) == length(vals) || error("Unequal number of variables and values")
            for (vn, val) in zip(vns, vals)
                DynamicPPL.setindex!(vi, val, vn)
            end
        end
    end
end

# Define a function which proposes a new set of parameters
"""
    proposewalk(rng, θ, Σ)

Draw a Multivariate Normal proposal centered at `θ` with covariance `Σ`.

# Arguments
- `rng::Random.AbstractRNG`: random number generator.
- `θ::AbstractVector`: current parameter vector (proposal mean).
- `Σ::AbstractMatrix`: covariance matrix for the proposal.

# Returns
- A vector containing the proposed parameters.
"""
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, Σ::AbstractMatrix)
    # Optimized: use direct Cholesky sampling instead of MvNormal constructor
    # This avoids allocating a new distribution object on every call
    n = length(θ)
    z = randn(rng, n)
    L = cholesky(Symmetric(Σ)).L
    return θ .+ L * z
end

# Overload for pre-factorized covariance (for caching scenarios)
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, L::LowerTriangular)
    # When Cholesky factorization is pre-computed and cached
    n = length(θ)
    z = randn(rng, n)
    return θ .+ L * z
end

# Define a function which proposes a new set of parameters
"""
    proposewalkram(rng, θ, S)

RAM-style proposal: sample a standard normal `U` and return `(θ + S*U, U)`.

# Arguments
- `rng::Random.AbstractRNG`: random number generator.
- `θ::AbstractVector`: current parameter vector.
- `S::LowerTriangular`: (Cholesky-like) factor such that Σ = S*S'.

# Returns
- `(proposal, U)` where `proposal` is the proposed parameter vector and `U` is
  the standard normal vector used to generate it. The returned `U` can be used
  for reversible jump / RAM adjustments.
"""
function proposewalkram(rng::Random.AbstractRNG, θ::AbstractVector, S::LowerTriangular)
    # Optimized: avoid allocating zeros vector and use randn directly
    U = randn(rng, length(θ))
    return θ + S * U, U
end

# Write a function which checks whether to accept a new set of parameters
"""
        mh_accept_logdens(rng, s_prime, s)

Decide whether to accept a Metropolis-Hastings proposal given log-densities.

# Arguments
- `rng::AbstractRNG`: random number generator.
- `s_prime::Real`: proposed log-density (log π(x')).
- `s::Real`: current log-density (log π(x)).

# Returns
- `Bool` indicating whether to accept the proposal.

# Notes
- Uses the `-randexp(rng) < min(1e-6, s_prime - s)` trick present in the
    original code. Consider replacing `1e-6` with `0.0` if numerical behaviour is
    acceptable — the current cutoff appears to guard against extreme values.
"""
mh_accept_logdens(rng::T, s_prime::T2, s::T2) where {T <:AbstractRNG, T2 <: Real} = -randexp(rng) < min(1e-6, s_prime - s)

# function mh_accept_tempered(
#     rng::AbstractRNG,
#     prop_varinfo::DynamicPPL.VarInfoOrThreadSafeVarInfo,
#     curr_varinfo::DynamicPPL.VarInfoOrThreadSafeVarInfo,
#     temp::Real,
# )   
#     logprior_prop= DynamicPPL.getlogprior(prop_varinfo)
#     logprior_curr= DynamicPPL.getlogprior(curr_varinfo)
# 
#     loglik_prop = DynamicPPL.getloglikelihood(prop_varinfo)
#     loglik_curr = DynamicPPL.getloglikelihood(curr_varinfo)
# 
#     logjac_prop = DynamicPPL.getlogjac(prop_varinfo)
#     logjac_curr = DynamicPPL.getlogjac(curr_varinfo)
# 
#     # tempered log posterior for proposal and current
#     logπ_prop = logprior_prop + temp * loglik_prop + logjac_prop
#     logπ_curr = logprior_curr + temp * loglik_curr + logjac_curr
# 
#     Δ = logπ_prop - logπ_curr
# 
#     # accept if log(u) < min(0, Δ)
#     accep = -randexp(rng) < min(1e-6, Δ)
#     return accep, accep ? logπ_prop : logπ_curr
# end

# Write a function which checks whether to accept a new set of parameters dr??
"""
    mh_dr_accept_logdens(rng, s_prime, s, s0)

Delayed-rejection acceptance logic operating on log-densities `s_prime`,
`s` and `s0` (log π(x'), log π(x), log π(x0)).

# Returns
- `(prob, accept)` where `prob` is a computed acceptance probability scalar
  (in some branches it may be 0 or 1) and `accept` is a Bool indicating
  whether the current RNG draw accepts the delayed-rejection step.

TODO: Add mathematical explanation and references for these computations.
"""
function mh_dr_accept_logdens(rng::T, s_prime::T2, s::T2, s0::T2) where {T <:AbstractRNG, T2 <: Real} 
    diff_delay_ll = exp(s_prime) - exp(s)
    diff_last_delay_ll = exp(s0) - exp(s)
    final_diff = exp(s_prime) - exp(s0)
    if(final_diff > 0)
        return 1, true
    end
    if(diff_delay_ll < 0)
        return 0, false
    end 
    if(diff_last_delay_ll < 0)
        return 1, true
    end
    prob = exp(log(diff_delay_ll) - log(diff_last_delay_ll))
    return  prob, -randexp(rng) < min(1e-6, log(diff_delay_ll) - log(diff_last_delay_ll))
end

"""
    getparams(model, t)

Return the parameter vector `θ` stored in the transition `t`.

This wrapper exists for compatibility with different versions/representations
of transition objects; it simply returns `t.θ` by default.
"""
function getparams(model::DynamicPPL.Model, t)
    return t.θ
end

"""
    _params_to_array(model, ts)

Convert a vector of transitions `ts` into a tuple `(names, vals)` where
`names` is an ordered collection of `VarName`s and `vals` is a matrix-like
collection of values (compatible with MCMCChains.jl).

This function uses `DynamicPPL.varname_and_value_leaves` to flatten structured
parameters into individual scalar entries so downstream tools can consume them.

TODO: Add an example showing how hierarchical parameter names are flattened.
"""
function _params_to_array(model::DynamicPPL.Model, ts::Vector)
    names_set = OrderedSet{VarName}()
    # Extract the parameter names and values from each transition.
    dicts = map(ts) do t
        # In general getparams returns a dict of VarName => values. We need to also
        # split it up into constituent elements using
        # `DynamicPPL.varname_and_value_leaves` because otherwise MCMCChains.jl
        # won't understand it.
        vals = getparams(model, t)
        nms_and_vs = if isempty(vals)
            Tuple{VarName,Any}[]
        else
            iters = map(DynamicPPL.varname_and_value_leaves, keys(vals), values(vals))
            mapreduce(collect, vcat, iters)
        end
        nms = map(first, nms_and_vs)
        vs = map(last, nms_and_vs)
        for nm in nms
            push!(names_set, nm)
        end
        # Convert the names and values to a single dictionary.
        return OrderedDict(zip(nms, vs))
    end
    names = collect(names_set)
    vals = [get(dicts[i], key, missing) for i in eachindex(dicts), key in names]

    return names, vals
end


struct AMGSContext{R<:AbstractRNG} <: DynamicPPL.AbstractContext
    rng::R
end

function DynamicPPL.tilde_assume!!(
    context::AMGSContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    # Allow MH to sample new variables from the prior if it's not already present in the
    # VarInfo.
    dispatch_ctx = if haskey(vi, vn)
        DynamicPPL.DefaultContext()
    else
        DynamicPPL.InitContext(context.rng, DynamicPPL.InitFromPrior())
    end
    return DynamicPPL.tilde_assume!!(dispatch_ctx, right, vn, vi)
end

function DynamicPPL.tilde_observe!!(
    ::AMGSContext, right::Distribution, left, vn::Union{VarName,Nothing}, vi::AbstractVarInfo
)
    return DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, vi)
end


# function _params_to_array(model::DynamicPPL.Model, ts::Vector)
#     names_set = OrderedSet{VarName}()
#     # Extract the parameter names and values from each transition.
#     dicts = map(ts) do t
#         nms_and_vs = Turing.Inference.getparams(model, t.params)
#         nms = map(first, nms_and_vs)
#         vs = map(last, nms_and_vs)
#         for nm in nms
#             push!(names_set, nm)
#         end
#         # Convert the names and values to a single dictionary.
#         return OrderedDict(zip(nms, vs))
#     end
#     names = collect(names_set)
#     vals = [
#         get(dicts[i], key, missing) for i in eachindex(dicts), (j, key) in enumerate(names)
#     ]
# 
#     return names, vals
# end
