using LinearAlgebra,Random,DelimitedFiles,TOML,SHA
import HighDimensionalModels as H
import Knockoffs as KJ
BLAS.set_num_threads(1)
root=ENV["HDM_REFERENCE"];base=dirname(root)
for case in 1:16
 S=readdlm(joinpath(base,"objective-suite-v1","case-$case","covariance.csv"),',',Float64);p=size(S,1)
 out=joinpath(root,"case-$case");mkpath(out);d=H.equicorrelated_s(S)
 Random.seed!(106200+case);Z=randn(p,p)
 Random.seed!(106200+case);external=KJ.condition(zeros(p,p),zeros(p),S,Diagonal(d))
 native=H.gaussian_knockoffs(zeros(p,p),zeros(p),S;s=d,noise=Z)
 for (name,value) in (("Z",Z),("native",native),("Knockoffs",external),("diagonal",d))
  writedlm(joinpath(out,"Julia-$name.csv"),value,',')
 end
end
metadata=Dict("Julia"=>string(VERSION),"Knockoffs"=>string(pkgversion(KJ)),"cpu"=>Sys.cpu_info()[1].model,
 "threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
 "backend_modelX_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(KJ)),"modelX.jl")))))
open(io->TOML.print(io,metadata),joinpath(root,"Julia-metadata.toml"),"w")
