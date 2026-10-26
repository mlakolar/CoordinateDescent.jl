include("study_common.jl")
sc=parse(Int,ARGS[1]);input=joinpath(STUDY,"inputs","sc$sc-shard0.bin")
out=joinpath(STUDY,"stopping-probe-v1");mkpath(out)
open(joinpath(out,"sc$sc.bin"),"w") do dst
 open(joinpath(out,"sc$sc.csv"),"w") do io
  println(io,"seed,method,seconds,converged,max_iterations,max_reported_residual")
  open(input) do src
   while (c=readcase(src))!==nothing
    X=sc==4 ? sparse(c.X) : c.X
    for j in 1:3
      w=c.W[:,j];amax=maximum(cumsum(sort(abs.(X'*c.y/c.n);rev=true))./cumsum(w))
      factors=collect(exp.(range(log(amax),log(.2amax);length=20)))
      elapsed=@elapsed z=H.SLOPEPath(X,c.y,factors;lambda=w,tol=1e-8,max_iter=100000)
      writerecord(dst,c,w*factors',vcat(hcat(z.βpath...),z.β0path'))
      println(io,join((c.seed,METHODS[j],elapsed,all(z.converged),maximum(z.iterations),maximum(z.kkt_residuals)),','));flush(io)
    end
   end
  end
 end
end
