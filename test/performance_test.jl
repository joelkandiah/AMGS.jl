using Turing
using Distributions
using Random
using LinearAlgebra
using AMGS
using BenchmarkTools

println("=== AMGS Performance Test ===\n")

# Set the seed for reproducibility
Random.seed!(1234)

# Define a simple 2D model for quick testing
@model function test_model_2d(σ², ρ)
    x ~ MvNormal([0.0, 0.0], [σ² ρ; ρ σ²])
end

# Define a higher-dimensional model for performance testing
@model function test_model_10d()
    # 10-dimensional multivariate normal
    Σ = diagm(ones(10)) + 0.1 * ones(10, 10)
    x ~ MvNormal(zeros(10), Σ)
end

println("Testing 2D model (correctness check)...")
model_2d = test_model_2d(1.0, 0.5)
myamgs = AMGS.AMGS_Sampler(0.234, 0.6)
spl_myamgs = DynamicPPL.Sampler(myamgs)

try
    # Quick run to verify it works
    chain_2d = sample(
        model_2d,
        spl_myamgs,
        MCMCSerial(),
        100,  # Small number for quick test
        1;
        chain_type = MCMCChains.Chains,
        num_warmup = 50,
        discard_init = 0,
    )
    println("✓ 2D model test passed\n")
catch e
    println("✗ 2D model test failed: $e\n")
    rethrow(e)
end

println("Testing 10D model (performance check)...")
model_10d = test_model_10d()
myamgs_10d = AMGS.AMGS_Sampler(0.234, 0.6)
spl_myamgs_10d = DynamicPPL.Sampler(myamgs_10d)

try
    # Benchmark the sampling
    println("Running benchmark (this may take a moment)...")
    result = @benchmark sample(
        $model_10d,
        $spl_myamgs_10d,
        MCMCSerial(),
        500,
        1;
        chain_type = MCMCChains.Chains,
        num_warmup = 100,
        discard_init = 0,
    )
    
    println("Benchmark results:")
    println("  Median time: $(median(result.times) / 1e9) seconds")
    println("  Mean time: $(mean(result.times) / 1e9) seconds")
    println("  Memory allocated: $(result.memory / 1024^2) MB")
    println("  Allocations: $(result.allocs)")
    println("\n✓ 10D model test completed\n")
catch e
    println("✗ 10D model test failed: $e\n")
    rethrow(e)
end

println("=== All performance tests completed ===")
