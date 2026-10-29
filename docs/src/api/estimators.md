# Estimators

[`CVMultiLassoPath`](@ref) tunes the multi-response row penalties with folds
shared by all responses.
[`dirty_multitask`](@ref), [`DirtyMultitaskPath`](@ref) and
[`CVDirtyMultitaskPath`](@ref) fit and tune separate shared and task-specific
coefficient components, with shared or task-specific designs.
[`transfer_multitask`](@ref), [`debias`](@ref), [`TransferMultitaskPath`](@ref)
and [`CVTransferMultitaskPath`](@ref) fit, correct and tune joint-sparse
transfer from source domains to a target.

```@docs
HighDimensionalModels
```

```@autodocs
Modules = [HighDimensionalModels]
Pages = ["src/lasso.jl", "src/penalties.jl", "src/glm.jl", "src/quantile.jl", "src/quantile_tuning.jl", "src/group.jl", "src/sparsegroup.jl", "src/multiresponse.jl", "src/multitask.jl", "src/cv.jl", "src/structured.jl", "src/hdm.jl", "src/lowrank.jl", "src/sparse_svd.jl"]
Private = true
```

```@docs
dirty_multitask
DirtyMultitaskPath
CVDirtyMultitaskPath
```

```@docs
transfer_multitask
debias
TransferMultitaskPath
CVTransferMultitaskPath
```
