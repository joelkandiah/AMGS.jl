using Revise
using Turing
using Distributions
using DynamicPPL
using AdvancedMH
using Random
using LinearAlgebra
using AMGS
using StableRNGs
using Dates

# ----------------------------------------------------------
# Model
# ----------------------------------------------------------
@model function demo(x, y)
    s² ~ Beta(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return s², m
end

model = demo(1.5, 2.0)

# ----------------------------------------------------------
# Samplers to test
# ----------------------------------------------------------
samplers = Dict(
    "AMGS" => AMGS.AMGS_Sampler(0.234, 0.6),
    "AMGS_Noise" => AMGS.AMGS_Sampler(0.234, 0.6, true, 1e-6),
    "Tempered" => AMGS.AMGS_Tempered_Sampler(0.234, 0.6, Val(false), 1e-6, 0.1),
    "Stopping_Tempered" => AMGS.AMGS_Stopping_Tempered_Sampler(
        0.234, 0.6, 10000, 0.22, 0.24, 200;
        use_noise_stabiliser = true, noise_scaling = 1e-6, temperature = 1.0
    ),
    "RAM" => AMGS.RAM_Sampler(0.234, 0.6),
    "Bayes_AMGS" => AMGS.Bayes_AMGS_Sampler(0.234, 100, 1, x -> Int(floor(0.5 * sqrt(2 * x)))),
    "Bayes_AMGS_Noise" => AMGS.Bayes_AMGS_Sampler(0.234, 100, 1, x -> Int(floor(0.5 * sqrt(2 * x))), true, 1e-5),
)

# ----------------------------------------------------------
# Benchmarking function
# ----------------------------------------------------------
function run_trial(model, sampler, seed)
    rng = StableRNG(seed)
    t_start = time()
    sample(
        model,
        sampler,
        MCMCSerial(),
        500, 1;
        chain_type = Chains,
        num_warmup = 100,
        thinning = 1,
        discard_initial = 100,
        progress = false,
        check_model = false,
        rng = rng
    )
    return time() - t_start
end

# ----------------------------------------------------------
# Run all benchmarks, 100 trials each
# ----------------------------------------------------------
outfile = "sampler_benchmarks_100_trials.txt"

open(outfile, "w") do io
    println(io, "Benchmark results generated: $(now())\n")

    for (name, sampler) in samplers
        println(io, "===================================================")
        println(io, "Sampler: $name")
        println(io, "Running 100 trials...\n")

        runtimes = Float64[]

        for trial in 1:100
            seed = 10_000 + trial
            elapsed = run_trial(model, sampler, seed)
            push!(runtimes, elapsed)
            println(io, "Trial $trial: $(round(elapsed; digits=6)) seconds")
        end

        avg = mean(runtimes)
        med = median(runtimes)
        sdv = std(runtimes)

        println(io, "\nSummary for $name:")
        println(io, "Mean runtime: $(round(avg; digits=6)) seconds")
        println(io, "Median runtime: $(round(med; digits=6)) seconds")
        println(io, "Std deviation: $(round(sdv; digits=6)) seconds\n")
    end
end

println("Wrote results to sampler_benchmarks_100_trials.txt")

