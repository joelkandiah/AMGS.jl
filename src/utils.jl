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

# Write a function which checks whether to accept a new set of parameters dr??
mh_dr_accept_logdens(rng::T, s_prime::T2, s::T2, s0::T2) where {T <:AbstractRNG, T2 <: Real} = -randexp(rng) < min(1e-6, log(exp(s) - exp(s_prime)) - log(exp(s0) - exp(s_prime)))

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
