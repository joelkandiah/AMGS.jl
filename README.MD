# AMGS

AMGS is a Julia package for Algebraic Multigrid Solvers. It provides efficient and scalable solvers for large sparse linear systems.

## Installation

To install AMGS, use the Julia package manager:

```julia
using Pkg
Pkg.add("AMGS")
```

## Using AMGS Sampler

Here is an example of how to use an AMGS sampler:

```julia
using Turing
using Random
using AMGS

# Define a simple model
@model function demo(x, y)
    s² ~ Beta(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return s², m
end

# Sample from the model
model_sample = demo(1.5, 2.0)

# Create an AMGS sampler
myamgs = AMGS.AMGS_Sampler(0.234, 0.6)
spl_myamgs = Turing.Sampler(myamgs)

# Perform inference
Random.seed!(1234)
chn = sample(model_sample, spl_myamgs, MCMCSerial(), 2000, 1; chain_type = Chains, num_warmup = 200, thinning = 1, discard_initial = 0)

# Plot the results
using StatsPlots
plot(chn)
```

