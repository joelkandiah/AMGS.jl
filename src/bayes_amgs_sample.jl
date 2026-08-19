"""
bayes_amgs_sample.jl

Bayesian adaptive Metropolis-within-Gibbs, following
https://onlinelibrary.wiley.com/doi/10.1111/anzs.12344 (Garthwaite, Fan &
Sisson style scale adaptation combined with a Bayesian covariance estimate).

Two things distinguish this sampler from base AMGS.

**The covariance is a Bayesian posterior mean, not a stochastic-approximation
average.** Placing an inverse-Wishart prior with `ν₀` degrees of freedom on the
target covariance and conditioning on the chain history gives a posterior mean
that is available in closed form. It is maintained incrementally over a
*sliding window* of the history `(f(n), n]`: `f` controls the oldest retained
draw and must grow slower than `n`, so that `n - f(n) → ∞` and the window keeps
widening while still discarding the badly-scaled burn-in draws. The update has
three cases:

1. the very first step, seeding from the two available points;
2. `f(n) == f(n-1)`, the window gained a point and lost none — a rank-one add;
3. `f(n) == f(n-1) + 1`, the window also dropped its oldest point — a rank-one
   add plus a rank-one removal, which is why the history is retained in a
   linked list.

**The scale is adapted multiplicatively with a restart rule.** `λ` follows

    λ ← max(λ_min, λ exp( (δ / (n_start + n)) (α - α*) ))

with the steplength constant `δ` set from the target acceptance rate and the
dimension. Whenever `λ` has moved more than `log 3` from the value at the last
restart, `n_start` is reset so that `n_start + n = 5/(α*(1-α*))`, which
re-inflates the gain and lets the scale move quickly again after a regime
change. The proposal covariance is `λ · c · S` with `c = 2.38²/d`, the classical
optimal random-walk scaling.

!!! note "Memory"
    The sliding-window update needs the draw leaving the window, so the chain
    history is kept in a linked list and grows linearly in the number of
    iterations.
"""

"""
    Bayes_AMGS_Sampler(acceptance_rate, ν₀, λ_min, f; use_noise_stabiliser=false, noise_scaling=1e-6)

Bayesian adaptive Metropolis-within-Gibbs sampler.

# Fields
- `acceptance_rate::Real`: target acceptance rate `α*`.
- `ν₀::Real`: degrees of freedom of the inverse-Wishart prior on the covariance.
  Larger values pull the estimate more strongly towards the initial covariance.
- `λ_min::Real`: floor on the multiplicative scale `λ`.
- `f`: the window function. Given the iteration `n` it returns the index of the
  oldest draw *excluded* from the window, so it must be non-decreasing, increase
  by at most 1 per step, and satisfy `n - f(n) → ∞`.
- `use_noise_stabiliser::Bool`, `noise_scaling::Real`: optional covariance noise
  stabiliser (see [`add_noise!`](@ref)).

# Example
```julia
# discard the first half of the history: window grows without bound
spl = Bayes_AMGS_Sampler(0.234, 5.0, 1e-6, n -> n ÷ 2)
```
"""
struct Bayes_AMGS_Sampler{
    T_accep<:Real,T_ν<:Real,T_λ<:Real,F,T_noise_scaling<:Real
} <: AbstractMCMC.AbstractSampler
    acceptance_rate::T_accep
    ν₀::T_ν
    λ_min::T_λ
    f::F
    use_noise_stabiliser::Bool
    noise_scaling::T_noise_scaling
end

function Bayes_AMGS_Sampler(
    acceptance_rate::T1,
    ν₀::T2,
    λ_min::T3,
    f;
    use_noise_stabiliser::Bool=false,
    noise_scaling::Real=1e-6,
) where {T1<:Real,T2<:Real,T3<:Real}
    0 < acceptance_rate < 1 ||
        throw(ArgumentError("acceptance_rate must lie strictly in (0, 1)"))
    λ_min > 0 || throw(ArgumentError("λ_min must be positive"))
    return Bayes_AMGS_Sampler(
        acceptance_rate, ν₀, λ_min, f, use_noise_stabiliser, noise_scaling
    )
end

"""
    BayesAMGSState

State for [`Bayes_AMGS_Sampler`](@ref).

# Fields
- `ldf`, `transition`, `θ`, `logp`: as in [`AMGSState`](@ref).
- `μ`: mean of the current history window (`X̄`).
- `kernel`: the Bayesian posterior-mean covariance `S`.
- `step`: iteration counter.
- `λ`, `λ_start`, `n_start`: multiplicative scale, its value at the last
  restart, and the offset that sets the current adaptation gain.
- `A`, `δ`: constants of the Garthwaite–Fan–Sisson steplength, derived from
  `α*` and the dimension.
- `d`: dimension of the parameter vector.
- `c`: the `2.38²/d` optimal random-walk scaling constant.
- `vec_X`: linked list of the chain history, needed to drop draws leaving the
  window.
"""
struct BayesAMGSState{
    TLDF<:DynamicPPL.LogDensityFunction,
    TTrans<:DynamicPPL.ParamsWithStats,
    Tθ<:AbstractVector,
    TKernel<:AbstractMatrix,
    TStep<:Integer,
    Tλ<:Real,
    Tn_start<:Real,
    TA<:Real,
    Tδ<:Real,
    Tc<:Real,
    TVecX,
    TL<:Real,
} <: AbstractAMGSState
    ldf::TLDF
    transition::TTrans
    θ::Tθ
    logp::TL
    μ::Tθ
    kernel::TKernel
    step::TStep
    λ::Tλ
    λ_start::Tλ
    n_start::Tn_start
    A::TA
    δ::Tδ
    d::Int
    c::Tc
    vec_X::TVecX
    θ_prop::Tθ
    z_cache::Tθ
end

"""
    get_index_ll(idx, vec_X)

Return the `idx`-th element (1-based) of the linked list `vec_X`.

Used to reach the draw leaving the history window, which sits a known distance
from the head.
"""
function get_index_ll(idx::Int, vec_X::LinkedList{T}) where {T}
    idx < 1 && throw(ArgumentError("Index must be greater than 0"))
    idx == 1 && return vec_X.head
    tmp = vec_X
    for _ in 1:(idx - 1)
        _, tmp = iterate(tmp)
    end
    return tmp.head
end

"""
    bayes_constants(acceptance_rate, d)

Derive the Garthwaite–Fan–Sisson steplength constants from the target
acceptance rate and the dimension.

`A` is the standard normal upper quantile at `α*/2`; `δ` is the resulting
steplength constant, and `n_start = 5/(α*(1-α*))` sets the initial adaptation
gain. Returns `(n_start, A, δ, c)` with `c = 2.38²/d`.
"""
function bayes_constants(a::Real, d::Integer)
    c = 2.38^2 / d
    n_start = 5 / (a * (1 - a))
    A = -quantile(Normal(0, 1), a * 0.5)
    δ = (1.0 - 1.0 / d) * (sqrt(2 * π) * exp((A^2) / 2) / (2 * A)) + 1.0 / (d * a * (1 - a))
    return n_start, A, δ, c
end

"""
    init_BayesAMGSState(ldf, transition, θ, logp, kernel, spl)

Build the initial Bayesian AMGS state, deriving the steplength constants from
the sampler's target acceptance rate and the problem dimension, and seeding the
history list with the starting point.
"""
function init_BayesAMGSState(
    ldf::DynamicPPL.LogDensityFunction,
    transition::DynamicPPL.ParamsWithStats,
    θ::AbstractVector,
    logp::Real,
    kernel::AbstractMatrix,
    spl::Bayes_AMGS_Sampler,
)
    d = length(θ)
    n_start, A, δ, c = bayes_constants(spl.acceptance_rate, d)
    μ = copy(θ)
    vec_X = cons(copy(θ), nil(typeof(μ)))
    λ₀ = one(Float64)
    return BayesAMGSState(
        ldf,
        transition,
        θ,
        logp,
        μ,
        kernel,
        1,
        λ₀,
        λ₀,      # λ_start
        n_start,
        A,
        δ,
        d,
        c,
        vec_X,
        similar(θ),
        similar(θ),
    )
end

"""
    update_kernel(rng, state, spl, prob)

Update the Bayesian covariance estimate over the sliding history window and
adapt the multiplicative scale `λ`.

Returns `(S, X̄, λ, λ_start, n_start, vec_X)`. Unlike the other samplers this
update is functional rather than in-place: the sliding-window recursions are
expressed as rank-one corrections to whole matrices, so there is no in-place
form that avoids the temporaries.
"""
function update_kernel(
    rng::AbstractRNG, state::BayesAMGSState, spl::Bayes_AMGS_Sampler, prob::Real
)
    S = state.kernel
    step = state.step
    X_n = copy(state.θ)              # current position (post accept/reject)
    X_n1 = state.vec_X.head          # previous position
    vec_X = cons(X_n, state.vec_X)

    μ = state.μ
    ν = spl.ν₀
    d = state.d
    n = size(X_n, 1)

    f_step = spl.f(step)
    # `f(0)` is never needed: step 1 takes the seeding branch.
    f_prev_step = step > 1 ? spl.f(step - 1) : f_step

    X̄ = μ
    if step == 1
        # Seed the estimate from the two points available so far.
        X̄ = (X_n .+ X_n1) ./ 2.0
        S = (1.0 / (ν + d + 3.0)) .*
            ((X_n1 * X_n1') .+ (X_n * X_n') .- 2.0 .* (X̄ * X̄') .+ (ν + d + 1.0) .* S)
    elseif f_step == f_prev_step
        # Window gained a point and lost none: rank-one add.
        w = step - f_step
        X̄ = (w / (w + 1.0)) .* μ .+ (1.0 / (w + 1.0)) .* X_n
        S = (1.0 / (w + ν + d + 2.0)) .*
            ((w + ν + d + 1.0) .* S .+ (X_n * X_n') .+ w .* (μ * μ') .-
             (w + 1.0) .* (X̄ * X̄'))
    elseif f_step == f_prev_step + 1
        # Window slid: add the new point and drop the oldest.
        w = step - f_step
        Xfn1 = get_index_ll(w + 2, vec_X)
        X̄ = μ .+ (1.0 / (w + 1)) .* (X_n .- Xfn1)
        S = S .+ (1.0 / (w + ν + d + 2.0)) .*
            ((X_n * X_n') .- (Xfn1 * Xfn1') .+ (w + 1.0) .* ((μ * μ') .- (X̄ * X̄')))
    else
        throw(
            ArgumentError(
                "the window function `f` must increase by at most 1 per step; " *
                "f($(step)) = $(f_step) but f($(step - 1)) = $(f_prev_step)",
            ),
        )
    end

    S = copy(S)
    add_noise!(rng, S, spl.noise_scaling, n, spl.use_noise_stabiliser)

    # Multiplicative scale adaptation with the restart rule.
    λ = max(
        spl.λ_min,
        state.λ * exp((state.δ / (state.n_start + step)) * (prob - spl.acceptance_rate)),
    )
    λ_start = state.λ_start
    n_start = state.n_start
    if abs(λ - λ_start) > log(3)
        λ_start = λ
        n_start = (5.0 / (spl.acceptance_rate * (1.0 - spl.acceptance_rate))) - step
    end

    return S, X̄, λ, λ_start, n_start, vec_X
end

####
#### Sampler interface
####

Turing.Inference.isgibbscomponent(::Bayes_AMGS_Sampler) = true
Turing.Inference.init_strategy(::Bayes_AMGS_Sampler) = DynamicPPL.InitFromPrior()

function _bayes_initial_state(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::Bayes_AMGS_Sampler, initial_params
)
    ldf = build_ldf(model, rng, initial_params)
    θ = initial_theta(ldf)
    logp = LogDensityProblems.logdensity(ldf, θ)
    transition = DynamicPPL.ParamsWithStats(θ, ldf, (; accepted=true))
    return init_BayesAMGSState(ldf, transition, θ, logp, initial_kernel(θ), spl)
end

"""
    _bayes_step!(rng, state, spl, Σ, discard_sample)

One iteration: propose, accept/reject, then update the windowed covariance and
the multiplicative scale.
"""
function _bayes_step!(
    rng::AbstractRNG,
    state::BayesAMGSState,
    spl::Bayes_AMGS_Sampler,
    Σ::AbstractMatrix,
    scale::Real,
    discard_sample::Bool,
)
    accep, logp, prob = metropolis_move!(rng, state, Σ, scale)
    S, X̄, λ, λ_start, n_start, vec_X = update_kernel(rng, state, spl, prob)

    moved = BayesAMGSState(
        state.ldf,
        state.transition,
        state.θ,
        logp,
        X̄,
        S,
        state.step + 1,
        λ,
        λ_start,
        n_start,
        state.A,
        state.δ,
        state.d,
        state.c,
        vec_X,
        state.θ_prop,
        state.z_cache,
    )

    # Keep the transition cache consistent with the position regardless of
    # `discard_sample`, which only suppresses what is returned.
    transition = accep ? make_transition(moved, true) : reuse_transition(moved, false)

    new_state = BayesAMGSState(
        moved.ldf,
        transition,
        moved.θ,
        moved.logp,
        moved.μ,
        moved.kernel,
        moved.step,
        moved.λ,
        moved.λ_start,
        moved.n_start,
        moved.A,
        moved.δ,
        moved.d,
        moved.c,
        moved.vec_X,
        moved.θ_prop,
        moved.z_cache,
    )

    return (discard_sample ? nothing : transition), new_state
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _bayes_initial_state(rng, model, spl, initial_params)
    return _bayes_step!(rng, state, spl, state.kernel, state.c, discard_sample)
end

function AbstractMCMC.step_warmup(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler,
    state::BayesAMGSState;
    discard_sample=false,
    kwargs...,
)
    return _bayes_step!(rng, state, spl, state.kernel, state.λ * state.c, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler;
    initial_params=Turing.Inference.init_strategy(spl),
    discard_sample=false,
    kwargs...,
)
    state = _bayes_initial_state(rng, model, spl, initial_params)
    return _bayes_step!(rng, state, spl, state.kernel, state.c, discard_sample)
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    spl::Bayes_AMGS_Sampler,
    state::BayesAMGSState;
    discard_sample=false,
    kwargs...,
)
    return _bayes_step!(rng, state, spl, state.kernel, state.λ * state.c, discard_sample)
end

####
#### Gibbs interface
####

function Turing.Inference.gibbs_update_state!!(
    ::Bayes_AMGS_Sampler,
    state::BayesAMGSState,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    new_ldf, θ, logp, transition = gibbs_rebuild(state, model, global_vals)
    return BayesAMGSState(
        new_ldf,
        transition,
        θ,
        logp,
        state.μ,
        state.kernel,
        state.step,
        state.λ,
        state.λ_start,
        state.n_start,
        state.A,
        state.δ,
        state.d,
        state.c,
        state.vec_X,
        state.θ_prop,
        state.z_cache,
    )
end

getlogevidence(transitions, sampler::Bayes_AMGS_Sampler, state) = missing
