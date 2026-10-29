# Solutions and StatsAPI

[`CVMultiLassoPath`](@ref) supports `coef` and `predict` at `λ1se` or `λmin`.
[`DirtyMultitaskSolution`](@ref) returns total coefficients and a canonical
component split, with a recomputed `kkt` residual and a relative duality `gap`.
Dirty CV uses its minimum-error `(ratio, λ_specific)` fit; it has no 1se rule.
[`TransferMultitaskSolution`](@ref) reports the fused or corrected target
slopes with separate fused and correction certificates; `domain = ℓ` selects a
domain block in `coef` and `predict`.

```@autodocs
Modules = [HighDimensionalModels]
Pages = ["src/statsapi.jl", "src/formula.jl"]
Private = false
```

```@docs
DirtyMultitaskSolution
coef(::DirtyMultitaskSolution)
SparseNuclearSolution
LowRankPlusSparseSolution
```

```@docs
TransferMultitaskSolution
coef(::TransferMultitaskSolution)
```
