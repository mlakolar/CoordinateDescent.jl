using CoordinateDescent, Random, LinearAlgebra, Statistics, Test
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","..","..","test","inference.jl"))
include(joinpath(@__DIR__,"..","..","..","test","inference_golden.jl"))
include(joinpath(@__DIR__,"..","..","..","test","inference_tuning.jl"))
