######################################################################
#
#   Models fitted from a formula and a table (StatsModels extension)
#
######################################################################

"""
    FormulaFit

A model fitted from a formula and a table. Loading StatsModels.jl adds formula
methods to the estimators, which return it:

```julia
using CoordinateDescent, StatsModels
path = fit(LassoPath, @formula(y ~ x1 + x2 + log(x3) + region), data)
sol = lasso(@formula(y ~ x1 + x2 + region), data, 0.1; standardizeX = true)
grp = groupLasso(@formula(y ~ x1 + region + x2 & region), data, 0.05)   # a group per term
```

The formula methods are `fit` with [`LassoPath`](@ref), [`CVLassoPath`](@ref),
[`GLMLassoPath`](@ref) and [`GroupLassoPath`](@ref), and [`lasso`](@ref),
[`sqrtLasso`](@ref), [`adaptiveLasso`](@ref), [`glmLasso`](@ref),
[`groupLasso`](@ref) and [`rlasso`](@ref) with the formula and the table in
place of `X` and `y`. The other arguments and keywords are those of the
estimator, with these differences:

- The formula sets the intercept: `y ~ x1 + x2` fits one, `y ~ 0 + x1 + x2`
  does not, and the keyword `intercept` is an error.
- Categorical variables (strings, or `CategoricalArray`s) become dummy columns,
  coded by `contrasts`, a `Dict` from variable names to StatsModels contrasts.
  The lasso penalizes each dummy column; the group lasso takes the columns of
  each term as a group unless `groups` is given.
- `weights` and `offset`, where the estimator takes them, may also be names of
  columns of the table.
- Rows with a missing value in a variable of the model are left out.

`model` is the fitted model, a [`LassoPath`](@ref), [`LassoSolution`](@ref) and
so on, and `formula` the formula with the schema of the data applied.
[`coef`](@ref), [`nobs`](@ref), [`dof`](@ref), [`deviance`](@ref), `fitted`,
[`residuals`](@ref), `stderror`, `vcov` and `confint` forward to the model, with
their arguments: `coef(m, λ)` for a path. `coefnames(m)` names the coefficients
in the order of `coef(m)`, the intercept first, and `responsename(m)` the
response. `predict(m, newdata; kwargs...)` builds the columns of the table
`newdata` by the formula and predicts with the model.
"""
struct FormulaFit{M, F}
  model::M
  formula::F
  coefnames::Vector{String}
  responsename::String
  columns::Vector{Int}      # the columns of the formula's model matrix that the model was fitted on
end

for f in (:coef, :nobs, :dof, :deviance, :nulldeviance, :fitted, :residuals, :stderror, :vcov, :confint)
  @eval StatsAPI.$f(m::FormulaFit, args...; kwargs...) = StatsAPI.$f(m.model, args...; kwargs...)
end
"""
    coefnames(m::FormulaFit)
    responsename(m::FormulaFit)

The names of the coefficients of a model fitted from a formula, in the order of
`coef(m)` (the intercept first), and the name of its response.
"""
StatsAPI.coefnames(m::FormulaFit) = m.coefnames
StatsAPI.responsename(m::FormulaFit) = m.responsename

function Base.show(io::IO, m::FormulaFit)
  print(io, "FormulaFit of a ", nameof(typeof(m.model)), ": ", m.formula)
end

function Base.show(io::IO, ::MIME"text/plain", m::FormulaFit)
  println(io, "FormulaFit of a ", nameof(typeof(m.model)))
  println(io, "  formula: ", m.formula)
  print(io, "  ", length(m.coefnames), " coefficients")
end
