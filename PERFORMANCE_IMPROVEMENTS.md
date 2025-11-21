# Performance Improvements for AMGS.jl Gibbs Samplers

## Summary

This document describes the performance optimizations applied to the AMGS.jl (Adaptive Metropolis-within-Gibbs Samplers) package to improve runtime performance, particularly for the Gibbs samplers.

## Optimizations Applied

### 1. **Optimized Proposal Generation** (`utils.jl`)

#### `proposewalk()` function
- **Before**: Created a new `MvNormal` distribution object on every call
- **After**: Direct Cholesky decomposition and sampling
- **Impact**: Eliminates distribution object allocation overhead
- **Code change**:
  ```julia
  # Before
  return rand(rng, MvNormal(eltype(Σ).(θ), Σ))
  
  # After  
  n = length(θ)
  z = randn(rng, n)
  L = cholesky(Symmetric(Σ)).L
  return θ .+ L * z
  ```

#### `proposewalkram()` function
- **Before**: Allocated a zeros vector and created `MvNormal` with identity covariance
- **After**: Direct `randn()` call for standard normal samples
- **Impact**: Eliminates unnecessary allocations
- **Code change**:
  ```julia
  # Before
  U = rand(rng, MvNormal(zeros(length(θ)), LinearAlgebra.I))
  
  # After
  U = randn(rng, length(θ))
  ```

### 2. **BLAS-Optimized Kernel Updates** (All sampler files)

#### Symmetric Rank-1 Update
- **Before**: Nested loops with O(n²) scalar operations
- **After**: BLAS `syr!` function for symmetric rank-1 updates
- **Impact**: 
  - Uses highly optimized BLAS routines (typically vectorized/multi-threaded)
  - Reduces computation from explicit nested loops to single BLAS call
  - Better cache locality and memory access patterns

#### Affected files:
1. `amgs_sample.jl`
2. `amgs_DA_sample.jl`
3. `amgs_tempered_sample.jl`
4. `amgs_tempered_stopping_sample.jl`
5. `amgs_switch_scaling_sample.jl`
6. `amgs_stopping_sample.jl`

#### Code change example:
```julia
# Before (nested loops)
@inbounds for i in 1:n
    Δi = state.Δ_cache[i]
    for j in i:n
        state.kernel[i,j] += η * (Δi * state.Δ_cache[j] - state.kernel[i,j])
        state.kernel[j,i] = state.kernel[i,j]  # preserve symmetry
    end
end

# After (BLAS)
state.kernel .*= (1.0 - η)
BLAS.syr!('U', η, state.Δ_cache, state.kernel)
@inbounds for j in 1:n
    for i in (j+1):n
        state.kernel[i,j] = state.kernel[j,i]
    end
end
```

### 3. **Optimized Noise Addition** (`amgs_stopping_sample.jl`)

- **Before**: Used `diagm()` to create diagonal matrix then added
- **After**: Direct diagonal element updates
- **Impact**: Eliminates temporary matrix allocation

## Expected Performance Improvements

### Theoretical Analysis

1. **Proposal Generation**: 
   - Reduction in allocations per proposal step
   - Faster sampling from multivariate normal distributions
   
2. **Kernel Updates**:
   - O(n²) → O(n²) complexity remains, but with BLAS optimizations:
     - Vectorization (SIMD instructions)
     - Better cache utilization
     - Potential multi-threading in BLAS
   - Expected speedup: **2-10x** for kernel updates depending on parameter dimension

3. **Overall Impact**:
   - For models with many parameters (n > 10), kernel updates dominate
   - Expected overall speedup: **1.5-5x** depending on model complexity
   - Larger improvements for higher-dimensional parameter spaces

### Benchmarking

A performance test has been added in `test/performance_test.jl` that tests:
- 2D model (correctness check)
- 10D model (performance benchmark)

## Technical Details

### BLAS.syr! Function
`syr!` performs a symmetric rank-1 update:
```
A := alpha*x*x**T + A
```

Where:
- `'U'` indicates upper triangular storage
- `alpha` is the scalar multiplier (η in our case)
- `x` is the vector (Δ_cache)
- `A` is the symmetric matrix (kernel)

### Cholesky Decomposition
For a positive definite matrix Σ:
```
Σ = L*L**T
```

Where L is lower triangular. To sample from N(μ, Σ):
```
x = μ + L*z,  where z ~ N(0, I)
```

## Compatibility

- All optimizations maintain numerical equivalence to original code
- Function signatures and interfaces unchanged
- Backward compatible with existing code using AMGS

## Future Optimization Opportunities

1. **Cholesky Caching**: Cache Cholesky factorizations across steps when kernel doesn't change much
2. **In-place Operations**: Further reduce allocations with more `!` functions
3. **StaticArrays**: For small fixed-size problems, use StaticArrays.jl
4. **Parallel Tempering**: Parallelize across chains for tempered samplers

## References

- BLAS documentation: https://www.netlib.org/blas/
- Julia LinearAlgebra: https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
- Original AMGS paper: https://link.springer.com/article/10.1007/s11222-008-9110-y
