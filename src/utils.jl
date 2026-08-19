"""
utils.jl

Shared helpers for the AMGS samplers: random-walk proposals and
Metropolis-Hastings acceptance rules.

All of these operate on the *linked* (unconstrained, real-valued) parameter
vector — every sampler in this package is a form of random-walk Metropolis, so
the parameters must live in ℝⁿ. Constrained model parameters are transformed
before they reach these functions, and the accompanying log-Jacobian is carried
in the target density (`getlogjoint_internal`).
"""

"""
    proposewalk(rng, θ, Σ)

Draw a Gaussian random-walk proposal centred at `θ` with covariance `Σ`.

Samples via the Cholesky factor directly (`θ + L z`, `Σ = L Lᵀ`) rather than
constructing an `MvNormal`, which avoids allocating a distribution object on
every call.

Returns the proposed parameter vector.
"""
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, Σ::AbstractMatrix)
    L = cholesky(Symmetric(Σ)).L
    return θ .+ L * randn(rng, length(θ))
end

"""
    proposewalk(rng, θ, L::LowerTriangular)

Overload for a pre-computed Cholesky factor, for callers that cache the
factorisation across steps.
"""
function proposewalk(rng::Random.AbstractRNG, θ::AbstractVector, L::LowerTriangular)
    return θ .+ L * randn(rng, length(θ))
end

"""
    proposewalkram(rng, θ, S)

Robust Adaptive Metropolis (RAM) proposal: draw `U ~ N(0, I)` and return
`(θ + S U, U)`.

`S` is the lower-triangular factor with `Σ = S Sᵀ`. The standard normal draw `U`
is returned alongside the proposal because the RAM rank-one update of `S`
(Vihola, 2012) is expressed in terms of `U`, not the proposal itself.
"""
function proposewalkram(rng::Random.AbstractRNG, θ::AbstractVector, S::LowerTriangular)
    U = randn(rng, length(θ))
    return θ + S * U, U
end

"""
    mh_accept_logdens(rng, s_prime, s)

Metropolis acceptance test for a **symmetric** proposal, given the target
log-densities `s_prime = log π(θ')` and `s = log π(θ)`.

Accepts with probability `min(1, exp(s_prime - s))`, implemented as
`-Exp(1) < min(0, Δ)`. The upper clamp is written as `1e-6` rather than `0`;
since `-randexp(rng)` is always strictly negative this is numerically identical
to clamping at `0` (any `Δ > 0` accepts with probability 1) while avoiding a
branch on the exact-zero case.

Returns `true` to accept the proposal.
"""
function mh_accept_logdens(rng::AbstractRNG, s_prime::Real, s::Real)
    return -randexp(rng) < min(1e-6, s_prime - s)
end

"""
    mh_dr_accept_logdens(rng, s_prime, s, s0)

Delayed-rejection (second-stage) acceptance test.

Given the log-densities of the second-stage proposal (`s_prime`), the current
state (`s`) and the already-rejected first-stage proposal (`s0`), the standard
delayed-rejection acceptance probability for symmetric proposals (Mira, 2001;
Tierney & Mira, 1999) is

    α₂ = min(1, [π(θ') - π(θ₀')] / [π(θ) - π(θ₀')])

where the bracketed differences are on the *density* scale and account for the
fact that the first-stage proposal was rejected.

Returns `(prob, accept)`, with `prob` the acceptance probability and `accept`
the realised decision.

!!! warning
    The differences are formed by exponentiating the log-densities, so this is
    prone to overflow/underflow for log-densities of large magnitude. It is only
    used by the pre-release delayed-rejection sampler.
"""
function mh_dr_accept_logdens(rng::AbstractRNG, s_prime::Real, s::Real, s0::Real)
    diff_delay_ll = exp(s_prime) - exp(s)
    diff_last_delay_ll = exp(s0) - exp(s)
    final_diff = exp(s_prime) - exp(s0)
    if final_diff > 0
        return 1, true
    end
    if diff_delay_ll < 0
        return 0, false
    end
    if diff_last_delay_ll < 0
        return 1, true
    end
    logratio = log(diff_delay_ll) - log(diff_last_delay_ll)
    return exp(logratio), -randexp(rng) < min(1e-6, logratio)
end
