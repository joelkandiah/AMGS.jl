# AMGS

AMGS is a Julia package for Adaptive metropolis hastings algorithms. 
## Installation

To install AMGS, use the Julia package manager:
**Not yet working, need to dev add for now**

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

For more details on the AMGS sampler, refer to [this paper](https://link.springer.com/article/10.1007/s11222-008-9110-y).

## Using Bayes AMGS Sampler

Here is an example of how to use a Bayes AMGS sampler:

```julia
# Create a Bayes AMGS sampler
mybayesamgs = AMGS.Bayes_AMGS_Sampler(0.234, 1.0, 0.1, x -> x ÷ 2)
spl_mybayesamgs = Turing.Sampler(mybayesamgs)

# Perform inference
Random.seed!(1234)
chn = sample(model_sample, spl_mybayesamgs, MCMCSerial(), 2000, 1; chain_type = Chains, num_warmup = 200, thinning = 1, discard_initial = 0)

# Plot the results
using StatsPlots
plot(chn)
```

For more details on the Bayes AMGS sampler, refer to [this paper](https://onlinelibrary.wiley.com/doi/10.1111/anzs.12344).

## Using AMGS Stopping Sampler

Here is an example of how to use an AMGS stopping sampler:

```julia
# Create an AMGS stopping sampler
myamgs_stopping = AMGS.AMGS_Stopping_Sampler(0.234, 0.6, 2000, 0.231, 0.237, 600)
spl_myamgs_stopping = Turing.Sampler(myamgs_stopping)

# Perform inference
Random.seed!(1234)
chn = sample(model_sample, spl_myamgs_stopping, MCMCSerial(), 2000, 1; chain_type = Chains, num_warmup = 200, thinning = 1, discard_initial = 0)

# Plot the results
using StatsPlots
plot(chn)
```

For more details on the AMGS stopping sampler, refer to [this paper](https://arxiv.org/abs/2408.04178).

## Using RAM Sampler

Here is an example of how to use a RAM sampler:

```julia
# Create a RAM sampler
myram = AMGS.RAM_Sampler(0.234, 0.6)
spl_myram = Turing.Sampler(myram)

# Perform inference
Random.seed!(1234)
chn = sample(model_sample, spl_myram, MCMCSerial(), 2000, 1; chain_type = Chains, num_warmup = 200, thinning = 1, discard_initial = 0)

# Plot the results
using StatsPlots
plot(chn)
```

For more details on the RAM sampler, refer to [this paper](https://arxiv.org/abs/1011.4381).

