ENV["CD81_IMPROVEMENTS_LIBRARY"]="1"
include(joinpath(@__DIR__,"updated_worker.jl"))
function adaptive_main()
rep=parse(Int,ARGS[1]);d=problem("inputs/medium-$rep")
out="results/adaptive/medium-$rep/julia";mkpath(out)
w=vec(readarr("inputs/medium-$rep/adaptive_weights.bin",d.p))
lambda_max=parse(Float64,strip(read("inputs/medium-$rep/adaptive_max.txt",String)))
function fit(which,frac)
  CD.sparse_reduced_rank(d.X,d.Y,d.r,frac*lambda_max;weights=w,block_solver=which,
    max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
end
for which in (:cd,:fista);fit(which,.1);end
if get(ENV,"CD81_PROFILE_ONLY","0")!="1"
open(joinpath(out,"julia.csv"),"w") do io
  println(io,join(FIELDS,','))
  for rd in 1:3,which in circshift([:cd,:fista],mod(rep+rd,2)),frac in (.5,.1,.02)
    GC.gc(false)
    seconds=@elapsed z=fit(which,frac)
    A,V=api_factors(d,frac,z.B;block_solver=which,weights=w,maximum=lambda_max)
    X=d.X.-mean(d.X;dims=1);Y=d.Y.-mean(d.Y;dims=1)
    E=X*A*V'-Y;C=X'Y/d.n;G=X'E*V/d.n;GV=E'X*A/d.n
    residual=max(CD._lr_group_kkt(A,G,frac*lambda_max,w)/max(1,norm(C*V)),
      norm(GV-V*((V'GV+GV'V)/2))/max(1,norm(Y'X*A/d.n)),norm(V'V-I))
    status=residual<=1e-6 ? "passed" : "accuracy_failed"
    name="cd_sparse_adaptive_"*string(which)
    println(io,join(("adaptive","julia","medium",rep,"sparse_adaptive",name,"fit",frac,rd,
      seconds,z.iterations,z.objective,NaN,residual,status,z.converged,Sys.maxrss()/1024,0,""),','))
    if rd==1
      for (tag,array) in (("bin",z.B),("A",A),("V",V))
        open(f->write(f,array),joinpath(out,"$name-$frac.$tag"),"w")
      end
    end
  end
end
else
  @assert isfile(joinpath(out,"julia.csv")) && countlines(joinpath(out,"julia.csv"))==19
end
for which in (:cd,:fista)
  Profile.clear();began=time();calls=0
  @profile while time()-began<.25 || calls==0
    fit(which,.1);calls+=1
  end
  open(joinpath(out,"profile-cd_sparse_adaptive_$which.txt"),"w") do io
    println(io,"rep=",rep," calls=",calls," samples=",length(Profile.fetch()))
    Profile.print(io;format=:flat,sortedby=:count,mincount=2)
  end
end

end
adaptive_main()
