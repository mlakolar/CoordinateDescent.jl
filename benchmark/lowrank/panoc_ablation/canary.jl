using HighDimensionalModels,Test,StatsAPI,LinearAlgebra,Random,Statistics,SparseArrays
BLAS.set_num_threads(1)
@test Base.get_extension(HighDimensionalModels,:HighDimensionalModelsProximalAlgorithmsExt)===nothing
@test_throws ArgumentError nuclear_regression(ones(3,2),ones(3,2),.1;solver=:panoc)
include(joinpath(dirname(pathof(HighDimensionalModels)),"..","test","lowrank.jl"))
