# AMGS.jl architecture review

Review of `src/` using the deep-module vocabulary from Matt Pocock's
`codebase-design` skill: **module**, **interface**, **depth**, **seam**,
**adapter**, **leverage**, **locality**, and the **deletion test**.

Scope: a read of the source. The Julia toolchain could not be installed in the
session that produced this review (the Julia installer and Pkg registry are
blocked by network policy), so **nothing here has been run**. Every claim is
checked against the code by reading and grep. Each proposed change needs the
existing test suite (`test/runtests.jl`) run before merging.

## Summary

The package has one good seam already and one that is missing:

- **Good:** `src/shared.jl` holds the common kernel machinery (target,
  `build_ldf`, proposal, Metropolis move, Gibbs rebuild). The target is already
  a seam: `build_ldf(model, rng, init, getlogdensity)` takes the density as an
  argument, and `TemperedLogJoint` is an adapter that fills it.
- **Missing:** the **adaptation rule** and the **stopping rule** are not seams.
  Each sampler file re-implements the whole AbstractMCMC lifecycle around its
  own adaptation. That gives 8 sampler files with nearly identical scaffolding,
  and the stopping logic is copied twice instead of wrapped once.

The fix is to make adaptation and stopping seams, so that a sampler is
"target + adaptation + optional stopping" and the lifecycle is written once.

## 1. Duplicated scaffolding in every sampler file (highest leverage)

Each of the 8 sampler files (`amgs_sample`, `amgs_stopping_sample`,
`amgs_tempered_sample`, `amgs_tempered_stopping_sample`, `amgs_DA_sample`,
`amgs_switch_scaling_sample`, `ram_sample`, `bayes_amgs_sample`) defines the
same set of pieces:

| Piece | Count | Notes |
|---|---|---|
| `struct …State` with ~11 fields | 8 | Only the adaptation fields differ |
| `init_…State` constructor | 8 | Allocates `similar(θ)` buffers |
| `_…initial_state` (build LDF, θ, logp, transition) | 8 | Same 5 lines in each |
| `_…step!` (propose, accept, cache transition, adapt) | 8 | Same skeleton |
| `step_warmup` ×2 and `step` ×2 | 8 × 4 | Identical bodies apart from the state call |
| `gibbs_update_state!!` | 8 | Rebuilds the full state positionally |
| `isgibbscomponent`, `init_strategy`, `getlogevidence` | 8 × 3 | Constants |

The Turing internal API is referenced in every one of these files (5 or 6
references per file, 41 in total across the 8 sampler files, counted by grep). Any change to Turing's Gibbs
interface, which already happened in the last commit, must be edited in 8
places. That is poor **locality**: the knowledge of "how an AMGS sampler plugs
into Turing" is spread out instead of living in one place.

**Proposed:** write the lifecycle once against an interface with three hooks.

```julia
# hooks each adaptation must provide
initial_adaptation(spl, θ)                        -> adaptation state
adapt(rng, spl, adaptation_state, move_info)      -> adaptation state
kernel(adaptation_state)                          -> Σ and scale for the next move
```

`step`, `step_warmup`, `gibbs_update_state!!` and the transition caching are
then defined once on an `AbstractAMGSSampler`. Each concrete sampler supplies
only its adaptation hooks.

Note `shared.jl` already has `adapt_scale(::Val{:RM|:DA|…}, …)` dispatch in
`amgs_switch_scaling_sample.jl`. That is the same seam in embryonic form, and
it shows the pattern works in this codebase.

## 2. Stopping is copied, not wrapped

`AMGS_Stopping_Sampler` duplicates `AMGS_Sampler`, and
`AMGS_Stopping_Tempered_Sampler` duplicates `AMGS_Tempered_Sampler`. The stopping
logic is the same in both (`block_update_state`, `check_end_adaptation`,
`_with_step`). The file `src/AMGS.jl` already carries a TODO that says so:

> Convert the stopping sampler into a wrapper for the other adaptive samplers
> -- it only needs the computed acceptance rate and the state returned from each
> step, so it could wrap `step` generically.

The reason the copy is awkward to remove: the stopping rule must **gate**
adaptation. Once it stops, `update_kernel!` is skipped. So a wrapper needs a
hook to say whether adaptation is still live:

```julia
adaptation_active(state) -> Bool
```

With that hook, `StoppingRule(inner_sampler)` becomes one generic
implementation instead of two copies. The four combinations (base / tempered ×
stopping / no stopping) fall out of two orthogonal seams, target and
stopping, instead of four hand-written types.

## 3. Positional state rebuilds are fragile

The stopping samplers rebuild their 15-field state **three times in one step**
(`moved`, the second `moved` after the kernel update, then `new_state`), by
listing every field positionally. Adding one field means editing every one of
those constructor calls, and a wrong field order does not raise an error (the
types line up). The base samplers have the same pattern once per step.

**Proposed:** either make the state a `mutable struct` with keyword
construction, or add one generic updater (for example `with(state; β = …)`)
that copies the fields by name. The generic updater is the lower-risk option
because it keeps the states immutable, which the samplers' allocation-free
design relies on.

## 4. Two proposal paths and dead helpers

- `propose!` (in `shared.jl`) is the in-place proposal the samplers use.
  `proposewalk` and `proposewalkram` (in `utils.jl`) allocate, and are used only
  by `test/benchmarks/`. Two proposal seams for one concept.
- `mh_accept_logdens` lives in `utils.jl` but is called by `shared.jl`. The
  `utils.jl` / `shared.jl` split is not a real seam: `utils.jl` is a grab-bag,
  not a module.
- **Dead code (deletion test passes):** `mh_dr_accept_logdens` in `utils.jl` has
  no callers. `src/pre-release/dr_amgs_sample.jl` is not `include`d by
  `AMGS.jl`, so it is not compiled and its complexity is not reachable from the
  package.

**Proposed:**
- Keep one proposal function (`propose!`) and have the benchmarks use it, or
  keep `proposewalk` only for the benchmark with a comment saying so.
- Delete `mh_dr_accept_logdens` (recoverable from git history if the
  delayed-rejection sampler is revived).
- Move `src/pre-release/` out of `src/` (for example into `attic/` or a branch)
  so the source tree shows only what is in the package.

## 5. Naming inconsistencies

These are small, but they add up when reading eight files side by side.

- `update_kernel!` (mutating, returns `(Σ, β)`) and `update_kernel` (no `!`,
  returns a new state) for the same role. Affects `ram_sample.jl` and
  `bayes_amgs_sample.jl`.
- Private helper prefixes differ per file: `_amgs_`, `_stopping_`, `_st_`,
  `_tempered_`, `_da_`, `_sgs_`, `_ram_`, `_bayes_`. Once the lifecycle is shared,
  these collapse to one prefix.
- State type names: `AMGSState`, `AMGSStoppingState`, `AMGSStoppingTemperedState`,
  `AMGSDAState`, `AMGSSGSState`, `RAMState`, `BayesAMGSState`. Exported sampler
  names follow a different convention (`AMGS_Stopping_Sampler`, `RAM_Sampler`).
  Fine to keep exports stable; the internal state names are free to change.

## 6. Module-level notes

- `AMGS.jl` has the stopping TODO attached to the `export` line, not to the
  stopping sampler. Move it next to the stopping code once that code is
  restructured.
- `Turing` is a direct dependency and is loaded whole (`using Turing`), but the
  package only uses `Turing.Inference.*` internals. Those internals are
  unstable (they changed in the last commit). Confining every `Turing.Inference`
  reference to one file (for example `src/gibbs.jl`) would make Turing upgrades a
  one-file change.
- `test/test_utils.jl` already has `test_sampler_contract`, which is the right
  idea: the interface is the test surface. It should be run over every variant
  (currently each variant has its own test file, so the contract is not
  enforced uniformly).

## Design options (design it twice)

For the sampler family as a whole, three ways to shape the interface:

**A. Generic sampler with composable parts.**
`AMGSSampler(target, adaptation; stopping = nothing)`. Old names are constructors
returning this type, so the public API is kept. Highest leverage: one
lifecycle, one state type. Biggest change, so it is best done in steps.

**B. Keep the sampler types, share the lifecycle through one interface.**
Each sampler type implements three adaptation hooks. A single `step` driver on
`AbstractAMGSSampler` does the rest. Smaller diff, less risk, and it is the
prerequisite for C and A.

**C. Stopping as a wrapper.**
`StoppingRule(inner)` gates adaptation through `adaptation_active`. Resolves the
existing TODO. Can be done on top of B.

**Recommendation:** B, then C, then A if the wrapper proves enough. Each step
leaves the public API unchanged and can be tested against the existing suite
before the next starts.

## Suggested order of work

1. Run the test suite on the current code and record the baseline (needs a
   Julia 1.10 environment with the `test/Project.toml` dependencies). Without a
   baseline, no refactor can be checked.
2. Delete dead code (`mh_dr_accept_logdens`, move `pre-release/` out of `src/`).
   Low risk, no behaviour change.
3. Confine the Turing internal calls to one `gibbs.jl` file. Mechanical move.
4. Replace the positional state rebuilds with a generic updater. Behaviour
   preserved, verified by the existing sampler tests.
5. Add the lifecycle driver and the three adaptation hooks (design B). Migrate
   one sampler at a time, starting with the base `AMGS_Sampler`, and keep its
   tests green before moving on.
6. Express stopping as a wrapper (design C) and delete the two copied stopping
   samplers.
7. Run `test_sampler_contract` over every variant.

## Tests to protect the refactor

The existing per-sampler tests (`test/test_amgs*.jl`, `test_ram.jl`,
`test_bayes_amgs.jl`) check the acceptance rate, covariance adaptation and
target correlation with fixed `StableRNG` seeds. Because they use fixed seeds,
a refactor that changes the order of random draws will change the numbers even
if it is correct. Before step 4, record the chain outputs for a fixed seed with
the current code, so a refactor can be checked for exact equality (not just
statistical agreement) and the tolerance question is settled.
