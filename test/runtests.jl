using CoordinateDescent
using Test
using Aqua

tests = [
  "coordinate_descent",
  "lasso",
  "cv",
  "penalties",
  "glm",
  "group",
  "multiresponse",
  "varying_coefficient_lasso"
]

@testset "CoordinateDescent" begin
  for t in tests
    f = "$t.jl"
    println("* running $f ...")
    include(f)
  end

  @testset "Aqua" begin
    Aqua.test_all(CoordinateDescent)
  end
end
