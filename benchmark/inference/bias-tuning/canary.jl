using HighDimensionalModels, Random, LinearAlgebra, Statistics, Test
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","..","..","test","inference.jl"))
include(joinpath(@__DIR__,"..","..","..","test","inference_golden.jl"))
include(joinpath(@__DIR__,"..","..","..","test","inference_tuning.jl"))
include(joinpath(@__DIR__,"..","..","..","test","inference_ldpe.jl"))
for (n,p,rho) in [(200,500,.3),(100,500,.9)]
 X=randn(Xoshiro(1),n,p)
 for j=2:p;X[:,j] .= rho.*X[:,j-1] .+ sqrt(1-rho^2).*X[:,j];end
 t=time_ns();P=nodewise_lasso(X,:ldpe;targets=1:10,threads=false,rng=Xoshiro(2))
 println("LDPE_CASE n=",n," p=",p," seconds=",(time_ns()-t)/1e9," converged=",all(P.converged)," lambda=",extrema(P.lambda)," eta=",extrema(P.bias_factor)," noise_ratio=",maximum(P.noise_factor./P.reference_noise))
 @test all(P.converged)
 @test all(P.noise_factor .<= 1.25P.reference_noise.+1e-8)
end
