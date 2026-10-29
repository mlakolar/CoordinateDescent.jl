# Solutions and StatsAPI

[`CVMultiLassoPath`](@ref) supports `coef` and `predict` at `λ1se` or `λmin`.
[`DirtyMultitaskSolution`](@ref) returns total coefficients and a canonical
component split, with a recomputed `kkt` residual and a relative duality `gap`.
Dirty CV uses its minimum-error `(ratio, λ_specific)` fit; it has no 1se rule.

```@autodocs
Modules = [HighDimensionalModels]
Pages = ["src/statsapi.jl", "src/formula.jl"]
Private = false
```

```@docs
DirtyMultitaskSolution
coef(::DirtyMultitaskSolution)
SparseNuclearSolution
```
