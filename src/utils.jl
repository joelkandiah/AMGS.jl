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