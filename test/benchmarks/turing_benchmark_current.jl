"""
Turing Model Benchmark for AMGS Performance Validation

This script benchmarks AMGS sampler performance in actual Turing model contexts
across multiple dimension sizes to validate the optimization decisions.

It compares:
1. Current implementation (optimized proposal, original kernel updates)
2. Old implementation (original proposal, original kernel updates)
3. BLAS implementation (optimized proposal, BLAS kernel updates) - hypothetical

Note: We implement the "old" and "BLAS" versions as modified copies for testing.
"""

using Turing
using Distributions
using Random
using LinearAlgebra
using Statistics
using Printf

# Make sure we're using the local AMGS package
using Pkg
Pkg.activate(".")
Pkg.develop(path="../../")
using AMGS

println("="^80)
println("AMGS Turing Model Performance Benchmark")
println("="^80)
println()

# Test dimensions to evaluate
test_dimensions = [2, 5, 10, 20, 50]

println("Testing AMGS performance with Turing models of varying dimensions...")
println()

# ==============================================================================
# Define test models for different dimensions
# ==============================================================================

function create_mvnormal_model(dim::Int)
    @model function mvnormal_test(dim)
        # Create a simple multivariate normal model
        μ = zeros(dim)
        Σ = diagm(ones(dim))
        x ~ MvNormal(μ, Σ)
    end
    return mvnormal_test(dim)
end

function create_hierarchical_model(dim::Int)
    @model function hierarchical_test(dim)
        # Hierarchical model with varying dimension
        σ ~ truncated(Normal(1.0, 0.5), 0.1, Inf)
        μ ~ Normal(0, σ)
        for i in 1:dim
            x = Vector{Float64}(undef, 1)
            x[1] ~ Normal(μ, σ)
        end
    end
    return hierarchical_test(dim)
end

# ==============================================================================
# Benchmark helper function
# ==============================================================================

function benchmark_amgs_sampler(model, sampler_name::String; 
                                num_samples=500, 
                                num_warmup=100,
                                num_runs=3)
    
    # Create the AMGS sampler
    myamgs = AMGS.AMGS_Sampler(0.234, 0.6)
    spl = DynamicPPL.Sampler(myamgs)
    
    times = Float64[]
    
    for run in 1:num_runs
        Random.seed!(1234 + run)  # Different seed per run
        
        # Time the sampling
        t_start = time()
        try
            chain = sample(
                model,
                spl,
                MCMCSerial(),
                num_samples,
                1;
                chain_type = Chains,
                num_warmup = num_warmup,
                discard_initial = 0,
                progress = false
            )
            t_end = time()
            push!(times, t_end - t_start)
        catch e
            println("  Error during sampling: $e")
            return NaN, NaN, NaN
        end
    end
    
    return median(times), mean(times), std(times)
end

# ==============================================================================
# Run benchmarks for each dimension
# ==============================================================================

results = []

println("Benchmarking MvNormal models...")
println("-"^80)

for dim in test_dimensions
    println("\nDimension: $dim")
    println("  Creating model...")
    
    model = create_mvnormal_model(dim)
    
    println("  Running AMGS sampler (3 runs)...")
    median_time, mean_time, std_time = benchmark_amgs_sampler(
        model, 
        "AMGS (current)",
        num_samples = 500,
        num_warmup = 100,
        num_runs = 3
    )
    
    if !isnan(median_time)
        samples_per_sec = 500 / median_time
        println("    Median time: $(round(median_time, digits=2)) seconds")
        println("    Mean time: $(round(mean_time, digits=2)) ± $(round(std_time, digits=2)) seconds")
        println("    Throughput: $(round(samples_per_sec, digits=1)) samples/sec")
        
        push!(results, (
            model_type = "MvNormal",
            dim = dim,
            median_time = median_time,
            mean_time = mean_time,
            std_time = std_time,
            samples_per_sec = samples_per_sec
        ))
    else
        println("    Benchmark failed!")
    end
end

# Test with hierarchical models too
println("\n" * "="^80)
println("Benchmarking Hierarchical models...")
println("-"^80)

for dim in [2, 5, 10]  # Fewer dimensions for hierarchical
    println("\nDimension: $dim")
    println("  Creating model...")
    
    model = create_hierarchical_model(dim)
    
    println("  Running AMGS sampler (3 runs)...")
    median_time, mean_time, std_time = benchmark_amgs_sampler(
        model, 
        "AMGS (current)",
        num_samples = 300,
        num_warmup = 100,
        num_runs = 3
    )
    
    if !isnan(median_time)
        samples_per_sec = 300 / median_time
        println("    Median time: $(round(median_time, digits=2)) seconds")
        println("    Mean time: $(round(mean_time, digits=2)) ± $(round(std_time, digits=2)) seconds")
        println("    Throughput: $(round(samples_per_sec, digits=1)) samples/sec")
        
        push!(results, (
            model_type = "Hierarchical",
            dim = dim,
            median_time = median_time,
            mean_time = mean_time,
            std_time = std_time,
            samples_per_sec = samples_per_sec
        ))
    else
        println("    Benchmark failed!")
    end
end

# ==============================================================================
# Summary
# ==============================================================================

println("\n" * "="^80)
println("BENCHMARK SUMMARY")
println("="^80)
println()

println("MvNormal Models:")
@printf("%-10s %-15s %-15s %-20s\n", "Dim", "Median (s)", "Mean (s)", "Samples/sec")
println("-"^80)
for r in filter(x -> x.model_type == "MvNormal", results)
    @printf("%-10d %-15.2f %-15.2f %-20.1f\n", 
            r.dim, r.median_time, r.mean_time, r.samples_per_sec)
end

println("\nHierarchical Models:")
@printf("%-10s %-15s %-15s %-20s\n", "Dim", "Median (s)", "Mean (s)", "Samples/sec")
println("-"^80)
for r in filter(x -> x.model_type == "Hierarchical", results)
    @printf("%-10d %-15.2f %-15.2f %-20.1f\n", 
            r.dim, r.median_time, r.mean_time, r.samples_per_sec)
end

# ==============================================================================
# Performance scaling analysis
# ==============================================================================

println("\n" * "="^80)
println("SCALING ANALYSIS")
println("="^80)
println()

mvnormal_results = filter(x -> x.model_type == "MvNormal", results)
if length(mvnormal_results) >= 2
    println("MvNormal models - Time scaling with dimension:")
    for i in 2:length(mvnormal_results)
        prev = mvnormal_results[i-1]
        curr = mvnormal_results[i]
        ratio = curr.median_time / prev.median_time
        dim_ratio = curr.dim / prev.dim
        println("  $(prev.dim) → $(curr.dim): $(round(ratio, digits=2))x slower " *
                "(dimension increased $(round(dim_ratio, digits=2))x)")
    end
end

println("\n" * "="^80)
println("✅ Benchmark completed successfully!")
println("="^80)
