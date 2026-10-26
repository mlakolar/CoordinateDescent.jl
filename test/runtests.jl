using HighDimensionalModels
using Test
using Aqua

tests = [
  "coordinate_descent",
  "selection",
  "slope_sparse",
  "slope_calibration",
  "selection_golden",
  "naming",
  "lasso",
  "cv",
  "structured",
  "structured_golden",
  "cv_glm",
  "penalties",
  "bounds",
  "glm",
  "quantile",
  "quantile_tuning",
  "glm_penalties",
  "group",
  "sparsegroup",
  "multiresponse",
  "lowrank",
  "hdm",
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
    # KernelSmoothing is a test dependency; on Julia 1.10, which ignores
    # [sources], CI adds it by URL, which also lists it in [deps]
    Aqua.test_all(HighDimensionalModels; stale_deps = (ignore = [:KernelSmoothing],))
  end
end
