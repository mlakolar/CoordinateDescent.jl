using HighDimensionalModels
using Test
using Aqua

tests = [
  "coordinate_descent",
  "selection",
  "knockoff_construction",
  "slope_sparse",
  "slope_calibration",
  "selection_golden",
  "naming",
  "lasso",
  "cv",
  "structured",
  "structured_golden",
  "structured_cv_masked",
  "cv_glm",
  "penalties",
  "bounds",
  "glm",
  "quantile",
  "quantile_tuning",
  "quantile_path_stop",
  "quantile_refit",
  "glm_penalties",
  "group",
  "sparsegroup",
  "multiresponse",
  "multitask",
  "multitask_penalties",
  "multitask_dirty",
  "lowrank",
  "hdm",
  "hdm_golden",
  "hdm_effect",
  "varying",
  "statsapi",
  "formula",
]

@testset "HighDimensionalModels" begin
  for t in tests
    f = "$t.jl"
    println("* running $f ...")
    include(f)
  end

  @testset "deprecated API" begin
    run(`$(Base.julia_cmd()) --startup-file=no --depwarn=yes --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "deprecated.jl"))`)
  end

  @testset "Aqua" begin
    Aqua.test_all(HighDimensionalModels)
  end
end
