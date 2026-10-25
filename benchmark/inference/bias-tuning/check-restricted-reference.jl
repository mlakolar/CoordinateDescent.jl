using CoordinateDescent, DelimitedFiles, Test, Random
X=readdlm("source-v3/test/data/inference/X.csv",',',Float64;skipstart=1)
levels=[.4,.3,.2,.15,.1,.07,.05,.03,.02,.01,.005,.001]
@testset "Independent residualized glmnet LDPE reference" begin
 for m in [0,4]
  P=nodewise_lasso(X,:ldpe;ldpe_unpenalized=m,lambda_path=levels,threads=false,options=CDOptions(;opt_tol=1e-11),rng=Xoshiro(1))
  R=readdlm("restricted-reference-diagnostics-$m.csv",',',Float64)
  @test Matrix(P.theta)≈readdlm("restricted-reference-theta-$m.csv",',',Float64) atol=1e-6
  @test P.lambda≈R[:,1] atol=1e-12
  @test P.bias_factor≈R[:,2] atol=1e-6
  @test P.noise_factor≈R[:,3] atol=1e-6
  @test P.reference_noise≈R[:,4] atol=1e-6
  @test P.bias_target_met==Bool.(R[:,5])
 end
end
