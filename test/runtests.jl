using CoordinateDescent
using Test
using Aqua

tests = [
  "coordinate_descent",
  "naming",
  "lasso",
  "cv",
  "cv_glm",
  "penalties",
  "bounds",
  "glm",
  "quantile",
  "glm_penalties",
  "group",
  "sparsegroup",
  "multiresponse",
  "hdm",
  "varying",
  "statsapi",
  "formula",
]

@testset "CoordinateDescent" begin
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
    Aqua.test_all(CoordinateDescent; stale_deps = (ignore = [:KernelSmoothing],))
  end
end
