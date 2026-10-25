# Each invocation is an independent data replication; run under Slurm.
using CoordinateDescent, LinearAlgebra, Random, Profile, Printf
BLAS.set_num_threads(1)
rep, out = parse(Int,ARGS[1]), ARGS[2]
mkpath(out)
open(joinpath(out,"metadata-rep$rep.txt"),"w") do io
  println(io,"Julia ",VERSION)
  println(io,"CPU ",Sys.CPU_NAME)
  println(io,"Julia threads ",Threads.nthreads(),"; BLAS threads ",BLAS.get_num_threads())
  println(io,"Slurm job ",get(ENV,"SLURM_JOB_ID","unset")," node ",get(ENV,"SLURMD_NODENAME","unset"))
end
function cases(rep,q)
  rng = Xoshiro(81000 + 100rep + q)
  n,p,r = 300,50,3
  X = randn(rng,n,p)
  U = zeros(p,r); U[1:10,:] = randn(rng,10,r)
  Y = X*U*randn(rng,r,q) + randn(rng,n,q)
  λmax = opnorm((X .- sum(X;dims=1)/n)'*(Y .- sum(Y;dims=1)/n))/n
  funcs = [
    "rank" => ()->reduced_rank(X,Y,3),
    "nuclear" => ()->nuclear_regression(X,Y,0.1λmax),
    "nuclear_path" => ()->LowRankPath(X,Y;λpath=λmax.*[1.01,0.5,0.2,0.1,0.05]),
    "sparse" => ()->sparse_reduced_rank(X,Y,3,0.1λmax),
    "sparse_svd" => ()->sparse_svd_regression(X,Y,3;nλ=8)
  ]
  X,Y,funcs
end
open(joinpath(out,"rep$rep.csv"),"w") do io
  println(io,"rep,q,method,round,seconds,bytes,converged,rank")
  for q in (10,100,1000)
    X,Y,funcs = cases(rep,q)
    foreach(x->last(x)(),funcs) # compile/warm up outside measurement
    for round in 1:3
      for j in circshift(collect(eachindex(funcs)),round+rep)
        name,f = funcs[j]
        GC.gc()
        m = @timed f()
        result = m.value
        sols = result isa LowRankPath ? result.solutions : result isa NamedTuple ? [result.solution] : [result]
        @printf(io,"%d,%d,%s,%d,%.9g,%d,%s,%d\n",rep,q,name,round,m.time,m.bytes,all(s->s.converged,sols),last(sols).rank)
        flush(io)
      end
    end
    # Independent profiles at every replication, separate from timing.
    for (name,f) in funcs
      Profile.clear()
      Profile.@profile for sample in 1:10
        f()
      end
      open(joinpath(out,"profile-rep$rep-q$q-$name.txt"),"w") do pf
        Profile.print(pf;format=:flat,sortedby=:count,mincount=2)
      end
    end
  end
end
