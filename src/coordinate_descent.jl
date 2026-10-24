"""
    coordinateDescent!(x, f, g, options=CDOptions(); rng=Random.default_rng(), kwargs...)
    solve!(x, f, g, options::CDOptions; rng=Random.default_rng(), kwargs...)

Minimize `f(x) + g(x)` by coordinate descent and store the solution in `x`. The
loss `f` is one of [`CDLeastSquaresLoss`](@ref), [`CDWeightedLSLoss`](@ref),
[`CDSqrtLassoLoss`](@ref) and [`CDQuadraticLoss`](@ref), a loss of ProximalBase
that ProximalSolvers's coordinate descent takes (`LeastSquaresLoss`, a
least-squares `GLMLoss`, `QuadraticFunction`, `KronQuadratic`, possibly scaled
as in `(1/n) * LeastSquaresLoss(y, X)`), a model of ProximalSolvers
(`ProximalSolvers.CDModel`), or a user-defined
[`CoordinateDifferentiableFunction`](@ref). The penalty `g` separates across
coordinates (`NormL1`, `ZeroFunction`, `ElasticNet`, `MCP`, `SCAD`, … from
ProximalBase, or another penalty, see below) and `x` is a `SparseIterate` or a
`SymmetricSparseIterate`, or `g` is a group lasso penalty `GroupNormL2` and `x`
an `AtomIterate` (see below).

The descent is the coordinate descent of ProximalSolvers (`ProximalSolvers.CDSolver`,
called through `ProximalSolvers.solve!`) with the options of [`CDOptions`](@ref).
It alternates between full passes over all coordinates and passes over the
nonzero coordinates only (the active set). It stops when a full pass is
followed by a pass in which no coordinate changes by more than
`options.optTol`, or after `options.maxIter` passes. In the second case `x` may
be far from the minimizer, and a warning is issued.

If `options.warmStart` is true, the descent starts from the supplied `x`.
Otherwise `x` is set to zero and, for `NormL1` and `GroupNormL2`, the problem
is solved along a geometric sequence of `options.numSteps` penalty levels that
decreases from the smallest `λ0` with zero solution to `g.λ0`, each one warm
started from the previous solution. Coordinates with zero penalty weight are
left out when finding that `λ0`. For a `SparseIterate` or an `AtomIterate`, each
penalty level is screened with the sequential strong rule, as in
[`LassoPath`](@ref).

The keyword arguments `kwargs` are fields of [`CDOptions`](@ref) that replace
those of `options`, as in `coordinateDescent!(x, f, g; optTol=1e-8)`. With
`order = :random`, the coordinates are visited in an order drawn from `rng`;
pass a seeded generator, such as `Xoshiro(1)`, for reproducible passes without
reseeding the global one.

`coordinateDescent!` returns `x`. `solve!` (a method of `ProximalSolvers.solve!`)
does the same and returns a `ProximalSolvers.CDResults`: the number of `passes`
(including those on the path of penalties when `warmStart` is false), whether
the stopping rule was met (`converged`), the largest scaled change of a
coordinate in the last pass (`maxchange`) and the `tolerance` it was compared
with, and the objective `minimum`. Passing a [`CDInfo`](@ref) as the fifth
argument of `coordinateDescent!`, to be filled in with this information, is
deprecated.

A dense `Vector` `x` is also accepted; it is converted to a `SparseIterate`, or
to an `AtomIterate` for `GroupNormL2`, and overwritten with the solution.

# Group lasso

For `g = GroupNormL2(λ0, groups, w)`, `g(x) = λ0 Σₖ wₖ ‖x[Gₖ]‖₂`, the
coordinates are the groups: `x` is an `AtomIterate` with the groups of `g`, such
as `AtomIterate(p, g.groups)`, and the active set is its set of active groups.
Each step minimizes over one group `G` the quadratic with curvature `L` that
majorizes `f` around `x`, plus the penalty: `x[G] ← (1 - λ0 wₖ / (L ‖u‖₂))₊ u`
with `u = x[G] - ∇_G f(x) / L` (groupwise majorization descent; Yang and Zou,
*A fast unified algorithm for solving group-lasso penalized learning
problems*, 2015). For [`CDLeastSquaresLoss`](@ref), with or without intercept,
`L` is the largest eigenvalue of `X[:, G]'X[:, G] / n` (centered), and for
[`CDQuadraticLoss`](@ref) that of `A[G, G]`; user-defined losses do not take
groups. The stopping rule takes the largest change of an entry in the pass, as
for single coordinates. The path starts from `maxₖ ‖∇_Gₖ f(0)‖₂ / wₖ` over the
penalized groups, and the strong rule keeps the groups with
`‖∇_G f‖₂ ≥ (2λ - λprev) wₖ`. Groups with zero weight are not penalized, and the
groups must cover all coordinates, so a coordinate that is not penalized needs
a group of its own with zero weight.

# Other penalties

A separable penalty `g(x) = Σₖ gₖ(xₖ)` of another type, a subtype of
`ProximalBase.AbstractFunction` with `ProximalBase.isseparable` true for it,
needs the proximal operator of one entry, `ProximalBase.prox_entry(g, v, k, γ)`,
which returns `argmin_z gₖ(z) + (z - v)² / (2γ)`. The built-in losses take this
proximal step on the quadratic in `x[k]`, so they accept such a penalty, and so
does a user-defined loss whose `descendCoordinate!` calls
`ProximalBase.prox_coordinate!` (see [`CoordinateDifferentiableFunction`](@ref)).
Such a penalty has no path of penalty levels: with `warmStart = false`, the
descent starts from zero and solves at `g` directly.

# Anderson acceleration

With `anderson = K` in [`CDOptions`](@ref), which needs `order = :cyclic`, the
passes over the active set are accelerated (Bertrand and Massias, *Anderson
acceleration of coordinate descent*, 2021): every `K + 1` passes over an
unchanged active set, the iterates are extrapolated and the extrapolated point
is kept if it lowers the objective. It needs a `SparseIterate` or a
`SymmetricSparseIterate`, and a loss whose value the model knows (the built-in
losses, or a callable user-defined loss). It pays off when coordinate descent
needs many passes, as at small penalties and with correlated columns. It keeps
the weakness of the cyclic order when the columns share a strong common factor,
where `order = :random` is much faster.

# Momentum

With momentum (`momentum` in [`CDOptions`](@ref)), each pass over the active
set starts from the extrapolated point `xₖ + βₖ (xₖ - xₖ₋₁)`, with Nesterov's
`βₖ = (k - 1) / (k + 2)`; the momentum restarts when the step of a pass points
against it (O'Donoghue and Candès, 2015), after a full pass, and when the
active set shrinks. The cache of a built-in loss moves with the point, at the
cost of a few vector operations per pass; a user-defined loss is initialized
at the extrapolated point. It needs a `SparseIterate` or a
`SymmetricSparseIterate`. Unlike Anderson acceleration, it works with the
random order, so it also helps when the columns share a common factor; with
the cyclic order it can stall.

By default (`momentum = nothing`) it is used with `order = :random`, a
`SparseIterate` and a built-in loss other than `KronQuadratic`, and not
otherwise. `momentum = true` asks for it with a user-defined loss as well,
which can pay off when initializing the loss costs less than a pass;
`momentum = false` turns it off.
"""
function coordinateDescent!(
  x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate, Vector{<:AbstractFloat}},
  f::Union{CDModel, AbstractFunction, CoordinateDifferentiableFunction},
  g::AbstractFunction,
  options::CDOptions=CDOptions();
  kwargs...)

  solve!(x, f, g, options; kwargs...)
  x
end

function coordinateDescent!(
  x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate, Vector{<:AbstractFloat}},
  f::Union{CDModel, AbstractFunction, CoordinateDifferentiableFunction},
  g::AbstractFunction,
  options::CDOptions,
  info::CDInfo;
  kwargs...)

  Base.depwarn("coordinateDescent!(x, f, g, options, info) is deprecated: use `r = solve!(x, f, g, options)`, " *
               "which returns the convergence information in a ProximalSolvers.CDResults " *
               "(r.passes, r.converged, r.maxchange, r.tolerance).", :coordinateDescent!)
  _record!(info, solve!(x, f, g, options; kwargs...))
  x
end

"""
    solve!(x, f, g, options::CDOptions; rng=Random.default_rng(), kwargs...)

The method of `ProximalSolvers.solve!` for the options of this package: the same
as [`coordinateDescent!`](@ref), but it returns a `ProximalSolvers.CDResults`,
with the number of `passes`, whether the stopping rule was met (`converged`),
the largest scaled change of a coordinate in the last pass (`maxchange`) and
the `tolerance` it was compared with.
"""
function ProximalSolvers.solve!(
  x::Union{SparseIterate, SymmetricSparseIterate, AtomIterate, Vector{<:AbstractFloat}},
  f::Union{CDModel, AbstractFunction, CoordinateDifferentiableFunction},
  g::AbstractFunction,
  options::CDOptions;
  rng::AbstractRNG=default_rng(),
  kwargs...)

  _solve!(x, _model(f, x), g, _options(options, kwargs); rng)
end

# what ProximalSolvers's solve! takes for f: a loss of ProximalBase or a model, as
# it is, and a user-defined loss through its model
_model(f::Union{CDModel, AbstractFunction}, x) = f
_model(f::CoordinateDifferentiableFunction, x) = _LossModel(f, x)
