"""
Extended Performance Testing for Additional Optimization Opportunities

This script tests the optimization suggestions from PERFORMANCE_IMPROVEMENTS.md:
1. Adaptive BLAS switching (use BLAS only when n > threshold)
2. Cholesky caching (cache factorizations when Σ doesn't change)
3. @simd annotations for nested loops
4. In-place operations

Focus: Dimensions 2-25 (priority) and up to 100 for completeness
"""

using Random
using LinearAlgebra
using Statistics
using Printf

println("="^80)
println("AMGS Extended Optimization Testing")
println("Testing additional optimization opportunities")
println("="^80)
println()

# Test dimensions - prioritize under 25
test_dimensions = [2, 5, 10, 15, 20, 25, 50, 100]
priority_dims = [2, 5, 10, 15, 20, 25]

println("Priority dimensions (< 25): ", priority_dims)
println("Extended test dimensions: ", test_dimensions)
println()

# ==============================================================================
# Benchmark helper
# ==============================================================================

function benchmark_function(f, args...; samples=1000, warmup=10)
    for _ in 1:warmup
        f(args...)
    end
    
    times = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        f(args...)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end
    
    return (
        median = median(times),
        mean = mean(times),
        std = std(times),
        min = minimum(times),
        max = maximum(times)
    )
end

# ==============================================================================
# Test 1: Cholesky Caching
# ==============================================================================

println("="^80)
println("TEST 1: Cholesky Caching")
println("Caching Cholesky factorization when Σ doesn't change")
println("="^80)
println()

# Current implementation (no caching)
function proposewalk_current(rng, θ, Σ)
    n = length(θ)
    z = randn(rng, n)
    L = cholesky(Symmetric(Σ)).L
    return θ .+ L * z
end

# Cached version (assumes Σ doesn't change between calls)
mutable struct CachedCholesky
    Σ::Matrix{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    
    function CachedCholesky(Σ)
        L = cholesky(Symmetric(Σ)).L
        new(Σ, L)
    end
end

function proposewalk_cached(rng, θ, cache::CachedCholesky)
    n = length(θ)
    z = randn(rng, n)
    return θ .+ cache.L * z
end

cholesky_results = []

println("Testing Cholesky caching (simulating repeated calls with same Σ):")
println("-"^80)

for n in test_dimensions
    is_priority = n in priority_dims
    marker = is_priority ? "★" : " "
    
    println("\n$(marker) Dimension n = $n $(is_priority ? "(PRIORITY)" : "")")
    
    θ = randn(n)
    Σ = Matrix{Float64}(I, n, n) .+ 0.1 .* randn(n, n)
    Σ = (Σ + Σ') / 2 + n * I
    rng = MersenneTwister(42)
    
    # Benchmark without caching (recomputes Cholesky each time)
    current_stats = benchmark_function(proposewalk_current, rng, θ, Σ; samples=1000)
    println("  Without caching: $(round(current_stats.median * 1e6, digits=2)) μs")
    
    # Benchmark with caching (Cholesky computed once)
    cache = CachedCholesky(Σ)
    cached_stats = benchmark_function(proposewalk_cached, rng, θ, cache; samples=1000)
    println("  With caching:    $(round(cached_stats.median * 1e6, digits=2)) μs")
    
    speedup = current_stats.median / cached_stats.median
    improvement = (speedup - 1) * 100
    println("  Speedup: $(round(speedup, digits=2))x ($(round(improvement, digits=1))% improvement)")
    
    push!(cholesky_results, (
        dim = n,
        priority = is_priority,
        without_cache = current_stats.median * 1e6,
        with_cache = cached_stats.median * 1e6,
        speedup = speedup
    ))
end

# ==============================================================================
# Test 2: @simd Annotations for Nested Loops
# ==============================================================================

println("\n" * "="^80)
println("TEST 2: @simd Annotations for Kernel Update Loops")
println("Testing if SIMD can speed up nested loops")
println("="^80)
println()

# Original nested loops
function update_kernel_original(Δ_cache, kernel, η)
    n = length(Δ_cache)
    kernel_copy = copy(kernel)
    
    @inbounds for i in 1:n
        Δi = Δ_cache[i]
        for j in i:n
            kernel_copy[i,j] += η * (Δi * Δ_cache[j] - kernel_copy[i,j])
            kernel_copy[j,i] = kernel_copy[i,j]
        end
    end
    
    return kernel_copy
end

# With @simd annotation
function update_kernel_simd(Δ_cache, kernel, η)
    n = length(Δ_cache)
    kernel_copy = copy(kernel)
    
    @inbounds for i in 1:n
        Δi = Δ_cache[i]
        @simd for j in i:n
            kernel_copy[i,j] += η * (Δi * Δ_cache[j] - kernel_copy[i,j])
        end
        for j in i:n
            kernel_copy[j,i] = kernel_copy[i,j]
        end
    end
    
    return kernel_copy
end

simd_results = []

println("Testing @simd annotations:")
println("-"^80)

for n in test_dimensions
    is_priority = n in priority_dims
    marker = is_priority ? "★" : " "
    
    println("\n$(marker) Dimension n = $n $(is_priority ? "(PRIORITY)" : "")")
    
    Δ_cache = randn(n)
    kernel = Matrix{Float64}(I, n, n) .+ 0.1 .* randn(n, n)
    kernel = (kernel + kernel') / 2
    η = 0.1
    
    # Benchmark original
    orig_stats = benchmark_function(update_kernel_original, Δ_cache, kernel, η; samples=1000)
    println("  Original:    $(round(orig_stats.median * 1e6, digits=2)) μs")
    
    # Benchmark with @simd
    simd_stats = benchmark_function(update_kernel_simd, Δ_cache, kernel, η; samples=1000)
    println("  With @simd:  $(round(simd_stats.median * 1e6, digits=2)) μs")
    
    speedup = orig_stats.median / simd_stats.median
    improvement = (speedup - 1) * 100
    println("  Speedup: $(round(speedup, digits=2))x ($(round(improvement, digits=1))% improvement)")
    
    # Verify correctness
    result_orig = update_kernel_original(Δ_cache, kernel, η)
    result_simd = update_kernel_simd(Δ_cache, kernel, η)
    max_diff = maximum(abs.(result_orig - result_simd))
    println("  Max difference: $(max_diff) $(max_diff < 1e-10 ? "✓" : "⚠️")")
    
    push!(simd_results, (
        dim = n,
        priority = is_priority,
        original = orig_stats.median * 1e6,
        with_simd = simd_stats.median * 1e6,
        speedup = speedup
    ))
end

# ==============================================================================
# Test 3: Adaptive BLAS Switching
# ==============================================================================

println("\n" * "="^80)
println("TEST 3: Adaptive BLAS Switching")
println("Use BLAS only when n > threshold")
println("="^80)
println()

# Adaptive implementation
function update_kernel_adaptive(Δ_cache, kernel, η; threshold=30)
    n = length(Δ_cache)
    kernel_copy = copy(kernel)
    
    if n >= threshold
        # Use BLAS for large n
        kernel_copy .*= (1.0 - η)
        BLAS.syr!('U', η, Δ_cache, kernel_copy)
        @inbounds for j in 1:n
            for i in (j+1):n
                kernel_copy[i,j] = kernel_copy[j,i]
            end
        end
    else
        # Use nested loops for small n
        @inbounds for i in 1:n
            Δi = Δ_cache[i]
            for j in i:n
                kernel_copy[i,j] += η * (Δi * Δ_cache[j] - kernel_copy[i,j])
                kernel_copy[j,i] = kernel_copy[i,j]
            end
        end
    end
    
    return kernel_copy
end

adaptive_results = []

println("Testing adaptive BLAS switching (threshold=30):")
println("-"^80)

for n in test_dimensions
    is_priority = n in priority_dims
    marker = is_priority ? "★" : " "
    
    println("\n$(marker) Dimension n = $n $(is_priority ? "(PRIORITY)" : "")")
    
    Δ_cache = randn(n)
    kernel = Matrix{Float64}(I, n, n) .+ 0.1 .* randn(n, n)
    kernel = (kernel + kernel') / 2
    η = 0.1
    
    # Benchmark original
    orig_stats = benchmark_function(update_kernel_original, Δ_cache, kernel, η; samples=1000)
    println("  Original (nested):  $(round(orig_stats.median * 1e6, digits=2)) μs")
    
    # Benchmark adaptive
    adaptive_stats = benchmark_function(update_kernel_adaptive, Δ_cache, kernel, η; samples=1000)
    println("  Adaptive:           $(round(adaptive_stats.median * 1e6, digits=2)) μs")
    
    speedup = orig_stats.median / adaptive_stats.median
    improvement = (speedup - 1) * 100
    println("  Speedup: $(round(speedup, digits=2))x ($(round(improvement, digits=1))% improvement)")
    
    which_impl = n >= 30 ? "BLAS" : "Loops"
    println("  Implementation used: $which_impl")
    
    push!(adaptive_results, (
        dim = n,
        priority = is_priority,
        original = orig_stats.median * 1e6,
        adaptive = adaptive_stats.median * 1e6,
        speedup = speedup
    ))
end

# ==============================================================================
# Summary Tables
# ==============================================================================

println("\n" * "="^80)
println("SUMMARY - PRIORITY DIMENSIONS (< 25)")
println("="^80)
println()

println("Cholesky Caching Results:")
@printf("%-10s %-15s %-15s %-12s %-15s\n", "Dim", "No Cache (μs)", "Cached (μs)", "Speedup", "Improvement")
println("-"^80)
for r in filter(x -> x.priority, cholesky_results)
    improvement = (r.speedup - 1) * 100
    @printf("%-10d %-15.2f %-15.2f %-12.2fx %-15.1f%%\n", 
            r.dim, r.without_cache, r.with_cache, r.speedup, improvement)
end

avg_speedup_chol = mean(r.speedup for r in filter(x -> x.priority, cholesky_results))
println("\nAverage speedup (priority dims): $(round(avg_speedup_chol, digits=2))x")

println("\n@simd Annotation Results:")
@printf("%-10s %-15s %-15s %-12s %-15s\n", "Dim", "Original (μs)", "@simd (μs)", "Speedup", "Improvement")
println("-"^80)
for r in filter(x -> x.priority, simd_results)
    improvement = (r.speedup - 1) * 100
    @printf("%-10d %-15.2f %-15.2f %-12.2fx %-15.1f%%\n", 
            r.dim, r.original, r.with_simd, r.speedup, improvement)
end

avg_speedup_simd = mean(r.speedup for r in filter(x -> x.priority, simd_results))
println("\nAverage speedup (priority dims): $(round(avg_speedup_simd, digits=2))x")

println("\nAdaptive BLAS Results:")
@printf("%-10s %-15s %-15s %-12s %-15s\n", "Dim", "Original (μs)", "Adaptive (μs)", "Speedup", "Improvement")
println("-"^80)
for r in filter(x -> x.priority, adaptive_results)
    improvement = (r.speedup - 1) * 100
    @printf("%-10d %-15.2f %-15.2f %-12.2fx %-15.1f%%\n", 
            r.dim, r.original, r.adaptive, r.speedup, improvement)
end

avg_speedup_adaptive = mean(r.speedup for r in filter(x -> x.priority, adaptive_results))
println("\nAverage speedup (priority dims): $(round(avg_speedup_adaptive, digits=2))x")

# ==============================================================================
# Recommendations
# ==============================================================================

println("\n" * "="^80)
println("RECOMMENDATIONS FOR PRIORITY DIMENSIONS (< 25)")
println("="^80)
println()

println("1. Cholesky Caching:")
if avg_speedup_chol > 1.2
    println("   ✅ RECOMMENDED - Average $(round((avg_speedup_chol-1)*100, digits=1))% improvement")
    println("   Note: Only beneficial when Σ doesn't change between calls")
    println("   Implementation: Could add optional caching in state")
elseif avg_speedup_chol > 1.0
    println("   ⚠️  MARGINAL - Average $(round((avg_speedup_chol-1)*100, digits=1))% improvement")
else
    println("   ❌ NOT RECOMMENDED - Performance regression")
end

println("\n2. @simd Annotations:")
if avg_speedup_simd > 1.1
    println("   ✅ RECOMMENDED - Average $(round((avg_speedup_simd-1)*100, digits=1))% improvement")
    println("   Implementation: Add @simd to inner loop in kernel updates")
elseif avg_speedup_simd > 1.0
    println("   ⚠️  MARGINAL - Average $(round((avg_speedup_simd-1)*100, digits=1))% improvement")
else
    println("   ❌ NOT RECOMMENDED - Performance regression")
end

println("\n3. Adaptive BLAS:")
if avg_speedup_adaptive > 1.05
    println("   ✅ RECOMMENDED - Average $(round((avg_speedup_adaptive-1)*100, digits=1))% improvement")
    println("   Implementation: Switch to BLAS at n=30 threshold")
elseif avg_speedup_adaptive >= 0.95
    println("   ⚠️  NEUTRAL - No significant impact ($(round((avg_speedup_adaptive-1)*100, digits=1))%)")
else
    println("   ❌ NOT RECOMMENDED - Performance regression")
end

println("\n" * "="^80)
println("Testing complete!")
println("="^80)
