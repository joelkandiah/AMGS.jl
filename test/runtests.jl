using Turing
using Distributions
using AdvancedMH
using Random
using LinearAlgebra
using MCMCChains

using Test: @test, @testset, @test_throws
using AMGS

#!!! IGNORE ALL TESTS COMPLETELY POINTLESS

@testset "AMGS" begin
    
    # Set the seed
    Random.seed!(1234)

    # Base.@kwdef struct StatesExtractor{A}
    #     states::A = Vector{Any}()
    # end
    # function (callback::StatesExtractor)(
    #     rng,
    #     model,
    #     sampler,
    #     sample,
    #     state,
    #     iteration;
    #     kwargs...,
    # )
    #     if iteration == 1
    #         empty!(callback.states)
    #     end
    
    #     push!(callback.states, state)
    # end


    @testset "AMGS_Sampler" begin

    # Construct simple model
        for (σ²) in [
            (10.0),  # should hit upper bound easily
            (0.01), # should hit lower bound easily
        ]
    
            ρ = σ² / 2.0

            @model function test_model(σ², ρ)
                x ~ MvNormal([0.0, 0.0], [σ² ρ; ρ σ²])
            end

            test_model_ex = test_model(σ², ρ)

            # callback = StatesExtractor()

            myamgs = AMGS.AMGS_Sampler(0.234, 0.6)
            spl_myamgs = Turing.Sampler(myamgs)

            num_warmup = 200
            discard_init = 100
            num_samples = 1000
            
            @test try
                chain = sample(
                    test_model_ex,
                    spl_myamgs,
                    MCMCSerial(),
                    num_samples,
                    1;
                    chain_type = MCMCChains.Chains,
                    num_warmup = num_warmup,
                    discard_init = discard_init,
                    # callback = callback,
                )
            true
            catch e
                false
            end

        end
    end
    #     kernels = getproperty.(callback.states, [:kernel])   
    #     eigval_min = map(minimum, eachrow(mapreduce(eigvals, hcat, kernels)))
    #     eigval_max = map(maximum, eachrow(mapreduce(eigvals, hcat, kernels)))
    # end

end