include("study_common.jl")
engine=ARGS[1];sc=parse(Int,ARGS[2]);shard=parse(Int,ARGS[3])
engine in ("native","julia") || error("engine")
input=joinpath(STUDY,"inputs","sc$sc-shard$shard.bin")
out=joinpath(STUDY,"fits","$engine-sc$sc-shard$shard");mkpath(dirname(out))
ispath(out*".done") && error("immutable completed output exists")
open(input) do src
  open(out*".bin","w") do dst
    open(out*".csv","w") do status
      println(status,"seed,method,seconds,reported_converged,iterations,residual,error")
      while (c=readcase(src))!==nothing
        X,y,xb,yb=centered(c)
        K=engine=="native" ? 4 : 3
        W=K==4 ? hcat(c.W,c.W[:,2]) : c.W
        B=fill(NaN,c.p+1,K)
        for j in 1:K
          method=j<=3 ? METHODS[j] : "automatic"
          t=time_ns();ok=false;it=-1;res=NaN;err=""
          try
            if engine=="native"
              z=j==4 ? H.slope(c.X,c.y;calibration=:gaussian,q=.1,sigma=c.sigma,intercept=c.intercept,
                standardize_x=false,tol=1e-9,max_iter=100000) : H.slope(c.X,c.y,W[:,j];
                intercept=c.intercept,standardize_x=false,tol=1e-9,max_iter=100000)
              B[:,j]=vcat(z.x,z.intercept);ok=z.converged;it=z.iterations;res=z.kkt_residual
            else
              z=SJ.slope(X,y;α=[1.0],λ=W[:,j],fit_intercept=false,centering=:none,scaling=:none,
                tol=1e-8,max_it=100000,max_clusters=c.p+1,dev_change_tol=0.0,dev_ratio_tol=1.0)
              length(z.coefficients)==1 || error("truncated")
              b=vec(Matrix(z.coefficients[1]));B[:,j]=vcat(b,yb-dot(xb,b));ok=true
            end
          catch e
            err=replace(sprint(showerror,e),','=>';','\n'=>' ')
          end
          println(status,join((c.seed,method,(time_ns()-t)/1e9,ok,it,res,err),','));flush(status)
        end
        writerecord(dst,c,W,B)
      end
    end
  end
end
provenance(out*".toml",engine,input)
write(out*".done","complete\n")
