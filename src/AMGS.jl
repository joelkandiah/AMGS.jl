"""
AMGS

Adaptive Metropolis-within-Gibbs Samplers (AMGS)

This module provides a collection of adaptive Metropolis-Hastings samplers
implemented to work with Turing / DynamicPPL and AbstractMCMC. The samplers
implement online covariance adaptation (similar to Haario et al. and
Robbins–Monro style updates) and a set of utility functions for proposing
and updating parameter VarInfo objects.

Features
- AMGS sampler: adaptive Metropolis-within-Gibbs with running mean and
	covariance adaptation
- Variants: delayed-rejection, tempered, data-augmentation and stopping
	wrappers (see the individual sampler files in `src/`)

Usage
```
using AMGS
# Use the samplers via Turing/AbstractMCMC APIs, e.g.:
# sampler = AMGS_Sampler(0.234, 1000)
# chain = sample(model, sampler, 1000)
```

Notes and TODOs
- Many docstrings in the individual sampler files include TODO placeholders
	where more mathematical detail and algorithm references should be added.
	Please expand the algorithm description below and in the function-level
	docstrings where indicated.

Algorithm / background (fill in)
- [x] Add a brief description of the adaptive algorithm, reference papers,
	and the meaning of the parameters `acceptance_rate`, `convergence_step`,
	`noise_scaling` and `use_noise_stabiliser`.
"""

module AMGS

using Turing
using AbstractMCMC
using Random
using DynamicPPL
using LinearAlgebra
using BangBang
using Accessors
using MCMCChains
using OrderedCollections
using DataStructures
using LoopVectorization


#TODO: Resolve the in place operations in the code (BangBang.jl)
#TODO: Resolve the type instability in the code
#TODO: Extend these samplers to use the Gibbs sampling style mechanism for block samplers (want to wrap these samplers)
#TODO: Convert the stopping sampler into a wrapper for the other adaptive samplers (it ony really needs the computed acceptance rate and the state from the returns from each step so perhaps just a wrapper on the step function)
# TODO update the documenting comments
export step, step_warmup, bundle_samples, isgibbscomponent, setparams_varinfo!!, drop_space, setparams!!, get_varinfo, _add_noise!

# Functions which use a few different MH samplers
export AMGS_Sampler, AMGS_DA_Sampler, AMGS_Tempered_Sampler_Wrapper, AMGS_Stopping_Tempered_Sampler_Wrapper, AMGS_Stopping_Sampler_Wrapper, AMGS_Tempered_Sampler, AMGS_Stopping_Sampler, AMGS_Stopping_Tempered_Sampler, RAM_Sampler, Bayes_AMGS_Sampler, DRAMGS_Sampler, new_AMGS_Sampler, StoppingSampler, TemperedSampler, StoppingSampler, TemperedSampler

const PROGRESS = Ref(true)
# Utility functions borrowed from Turing packages (credit their code)
include("utils.jl")

# Code for each differnt sampler
include("amgs_sample.jl")
include("amgs_wrappers.jl")
include("amgs_stopping_sample.jl")
include("amgs_tempered_sample.jl")
include("amgs_tempered_stopping_sample.jl")
include("amgs_DA_sample.jl")
#include("pre-release/dr_amgs_sample.jl")
include("pre-release/new_amgs_sample.jl")
include("ram_sample.jl")
include("bayes_amgs_sample.jl")

end
