using Test

include("test_utils.jl")

@testset "AMGS" begin
    include("test_amgs.jl")
    include("test_amgs_stopping.jl")
    include("test_amgs_tempered.jl")
    include("test_amgs_tempered_stopping.jl")
    include("test_amgs_da.jl")
    include("test_amgs_sgs.jl")
    include("test_ram.jl")
    include("test_bayes_amgs.jl")
end
