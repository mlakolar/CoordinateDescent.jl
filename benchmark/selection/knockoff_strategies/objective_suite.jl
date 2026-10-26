# Numerical execution under Slurm only. One independent matrix case per worker.
using LinearAlgebra, Random, DelimitedFiles, TOML, SHA
import HighDimensionalModels as H
import Knockoffs as KJ
BLAS.set_num_threads(1)
case=parse(Int,ARGS[1]); root=ENV["HDM_CONSTRUCTION_SUITE"]
specs=[(4,0.),(4,.4),(4,.8),(12,.4),(12,.8),(12,.95),(30,.4),(30,.8),
       (30,.95),(100,.4),(100,.8),(100,.95),(300,.4),(300,.8),(300,.95),(12,-.4)]
p,rho=specs[case]
out=joinpath(root,"case-$case");ispath(out) && error("immutable suite case already exists");mkpath(out)
R=[rho^abs(i-j) for i in 1:p,j in 1:p]
scales=exp.(range(log(.2),log(5.);length=p));S=R.*(scales*scales')
writedlm(joinpath(out,"covariance.csv"),S,',')
settings=[("package-default",(;niter=100,tol=1e-6,λmin=1e-6,robust=false,verbose=false)),
          ("provisional",(;niter=1000,tol=1e-10,λmin=1e-10,robust=false,verbose=false)),
          ("robust",(;niter=1000,tol=1e-10,λmin=1e-10,robust=true,verbose=false)),
          ("tighter",(;niter=5000,tol=1e-12,λmin=1e-12,robust=true,verbose=false))]
open(joinpath(out,"diagonals.csv"),"w") do io
  println(io,"source,strategy,setting,success,seconds,error,s")
  d=H.knockoff_diagonal(S)
  println(io,join(("adapter","equicorrelated","default",true,NaN,"",join(d.s,';')),','))
  for strategy in (:mvr,:maxent),(name,opts) in settings,source in (:direct,:adapter)
    t=time_ns();d=fill(NaN,p);ok=false;err=""
    try
      d=source===:direct ? KJ.solve_s(Symmetric(copy(R)),strategy;opts...).*diag(S) :
        H.knockoff_diagonal(S;strategy,construction_options=opts).s
      ok=true
    catch e
      err=replace(sprint(showerror,e),','=>';','\n'=>' ')
    end
    println(io,join((source,strategy,name,ok,(time_ns()-t)/1e9,err,join(d,';')),','));flush(io)
  end
end
metadata=Dict("case"=>case,"p"=>p,"rho"=>rho,"julia"=>string(VERSION),
 "cpu"=>Sys.cpu_info()[1].model,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
 "Knockoffs"=>string(pkgversion(KJ)),"backend_status"=>"unavailable",
 "backend_utilities_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(KJ)),"utilities.jl")))),
 "construction_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoff_construction.jl")))),
 "covariance_sha256"=>bytes2hex(sha256(read(joinpath(out,"covariance.csv")))))
open(io->TOML.print(io,metadata),joinpath(out,"metadata.toml"),"w")
println("OBJECTIVE_SUITE_DONE $case")
