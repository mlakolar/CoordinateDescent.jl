using HighDimensionalModels, HiGHS, Random, Statistics, LinearAlgebra, DelimitedFiles, Test
const HDM=HighDimensionalModels
BLAS.set_num_threads(1)
loadbin(name)=open(io->read!(io,Vector{Float64}(undef,filesize("reference/$name.bin")÷8)),"reference/$name.bin")
X=reshape(loadbin("X"),83,5);y=loadbin("y");o=loadbin("offset");U=reshape(loadbin("uniforms"),83,128)
w=[.7,1.,1.2,.8,2.];fl=[i%3+1 for i in 1:83];ls=[.3,.15,.05]
refs=readdlm("reference/calibration.csv",',',Any;skipstart=1)
@testset "independent R calibration and quantile CV reference" begin
 for row in eachrow(refs)
  tau=Float64(row[1]);standardized=lowercase(string(row[2]))=="true";key="$(tau)-$(standardized ? "TRUE" : "FALSE")"
  B=reshape(loadbin("$key-coef"),6,3);E=reshape(loadbin("$key-fold-errors"),3,3)
  cv=CVQuantileLassoPath(X,y,ls;tau,standardize_x=standardized,penalty_factors=w,offset=o,folds=fl,threads=false,tol=2e-6)
  @test maximum(abs.(coef(cv.path).-B)) <= 2e-5
  @test maximum(abs.(cv.fold_errors.-E)) <= 2e-6
  @test cv.λmin==ls[argmin(vec(sum(E .* [27,28,28];dims=1)))]
  effective=standardized ? w .* vec(sqrt.(mean((X.-mean(X;dims=1)).^2;dims=1))) : w
  stats=HDM._quantile_score_maxima(X,effective,tau,true,U)
  @test 2quantile(stats,.9) ≈ Float64(row[3]) atol=1e-12
 end
end
# Warm-up precedes timing. No R startup, input generation or profiling is timed.
for method in (:exact,:smooth)
 CVQuantileLassoPath(X,y,ls;folds=fl,threads=false,method,rng=Xoshiro(31))
end
quantile_lambda(X;nsim=1000,rng=Xoshiro(31))
mkpath("results")
rows=NamedTuple[]
for seed in 1501:1503
 rng=Xoshiro(seed);A=randn(rng,1000,40);z=.4 .+ A[:,1:5]*[1.,-.7,.4,.3,.1] .+ randn(rng,1000)
 grid=exp.(range(log(.3),log(.01);length=12));folds=[i%5+1 for i in 1:1000]
 for rep in 1:2
  for method in shuffle(Xoshiro(seed+rep),[:exact,:smooth])
   GC.gc();t=@timed CVQuantileLassoPath(A,z,grid;folds,method,threads=false,tol=2e-6,rng=Xoshiro(31))
   @test all(isfinite,t.value.cvm)
   @test length(dual_scores(t.value))==1000
   push!(rows,(seed=seed,rep=rep,method=string(method),seconds=t.time,lambda_min=t.value.λmin,lambda_1se=t.value.λ1se))
  end
  GC.gc();t=@timed quantile_lambda(A;nsim=1000,rng=Xoshiro(31))
  push!(rows,(seed=seed,rep=rep,method="calibration",seconds=t.time,lambda_min=t.value,lambda_1se=NaN))
 end
end
open("results/timings.csv","w") do io
 println(io,join(keys(first(rows)),','))
 for r in rows;println(io,join(values(r),','));end
end
open("results/timing-summary.md","w") do io
 println(io,"|Operation|Mean seconds|Input SD|Inputs × repetitions|\n|---|---:|---:|---:|")
 for method in ("exact","smooth","calibration")
  vals=[mean(r.seconds for r in rows if r.seed==seed && r.method==method) for seed in 1501:1503]
  println(io,"|$method|$(mean(vals))|$(std(vals))|3 × 2|")
 end
end
println("TUNING VALIDATION COMPLETE: 24 R reference assertions; 18 repeated measurements")
