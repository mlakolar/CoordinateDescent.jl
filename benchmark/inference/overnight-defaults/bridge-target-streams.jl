using HighDimensionalModels,Random,LinearAlgebra,DelimitedFiles,Test
BLAS.set_num_threads(1)
caseid,rs=ARGS;rep=parse(Int,rs);base=joinpath("competitor-data",caseid,rs)
X=readdlm(joinpath(base,"X.csv"),',',Float64);y=vec(readdlm(joinpath(base,"y.csv"),',',Float64));targets=Int.(vec(readdlm(joinpath(base,"targets.csv"),',',Float64)))
f=debiased_lasso(X,y;targets,threads=false,rng=Xoshiro(970000+rep));@test f.converged
ref=readdlm(joinpath("outputs","gaussian-all-target-time",caseid,rs*".ours_all_warm.csv"),',',Any;header=true)[1]
mb=ms=mp=ml=0.
for (a,j) in enumerate(targets)
 row=findfirst(k->Int(ref[k,2])==j,axes(ref,1))
 global mb=max(mb,abs(f.beta[a]-Float64(ref[row,4])))
 global ms=max(ms,abs(f.se[a]-Float64(ref[row,5])))
 global mp=max(mp,abs(f.pvalues[a]-Float64(ref[row,8])))
 global ml=max(ml,abs(f.projection.lambda[a]-Float64(ref[row,14])))
end
@test mb<=1e-10 && ms<=1e-10 && mp<=1e-10 && ml<=1e-10
root=joinpath("outputs","target-stream-bridge",caseid);mkpath(root)
open(joinpath(root,rs*".csv"),"w") do io
 println(io,"case,rep,targets,max_beta,max_se,max_pvalue,max_lambda")
 println(io,join((caseid,rep,length(targets),mb,ms,mp,ml),','))
end
