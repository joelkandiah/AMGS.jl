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
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, Σ::AbstractMatrix)
    return rand(rng, MvNormal(eltype(Σ).(θ), Σ))
end

# Define a function which proposes a new set of parameters
function proposewalkram(rng::Random.AbstractRNG, θ::AbstractVector, S::LowerTriangular)
    U = rand(rng, MvNormal(zeros(length(θ)), LinearAlgebra.I))
    return θ + S * U, U
end

# Write a function which checks whether to accept a new set of parameters
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

function getparams(model::DynamicPPL.Model, t)
    return t.θ
end

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
