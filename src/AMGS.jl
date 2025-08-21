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

#TODO: Resolve the in place operations in the code (BangBang.jl)
#TODO: Resolve the type instability in the code
#TODO: Extend these samplers to use the Gibbs sampling style mechanism for block samplers (want to wrap these samplers)
#TODO: Convert the stopping sampler into a wrapper for the other adaptive samplers (it ony really needs the computed acceptance rate and the state from the returns from each step so perhaps just a wrapper on the step function)
# TODO update the documenting comments
export step, step_warmup, bundle_samples, isgibbscomponent, setparams_varinfo!!, drop_space, setparams!!, get_varinfo

# Functions which use a few different MH samplers
export AMGS_Sampler, AMGS_Stopping_Sampler, RAM_Sampler, Bayes_AMGS_Sampler, DRAMGS_Sampler, new_AMGS_Sampler

export AMGSState

# Utility functions borrowed from Turing packages (credit their code)
include("utils.jl")

# Code for each differnt sampler
include("amgs_sample.jl")
include("amgs_stopping_sample.jl")
#include("pre-release/dr_amgs_sample.jl")
include("pre-release/new_amgs_sample.jl")
include("ram_sample.jl")
#include("bayes_amgs_sample.jl")

end
