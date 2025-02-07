module AMGS

using Turing
using AbstractMCMC
using AdvancedMH
using Random
using DynamicPPL
using LinearAlgebra
using BangBang
using Accessors
using MCMCChains
using OrderedCollections


export step, step_warmup, bundle_samples

# Functions which use a few different MH samplers
export AMGS_Sampler, AMGS_Stopping_Sampler, RAM_Sampler, Bayes_AMGS_Sampler

export AMGSState

# Utility functions borrowed from Turing packages (credit their code)
include("utils.jl")

# Code for each differnt sampler


include("amgs_sample.jl")
include("amgs_stopping_sample.jl")
include("ram_sample.jl")
include("bayes_amgs_sample.jl")


end # module AMGS
