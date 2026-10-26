include("study_common.jl")
using Profile
engine=ARGS[1];sc=parse(Int,ARGS[2]);rep=parse(Int,ARGS[3])
input=joinpath(STUDY,"inputs","sc$sc-shard$rep.bin")
out=joinpath(STUDY,"fits","$engine-sc$sc-shard$rep");mkpath(out*"-profiles")
ispath(out*".done") && error("completed runtime outputs exist")
resume=get(ENV,"HDM_RESUME","0")=="1"
existing=Set{Tuple{Int,String,String,String,Int}}()
existing_weights=Set{Tuple{Int,String,Int}}()
existing_witness=Set{Tuple{Int,String,String}}()
if resume
    for line in Iterators.drop(eachline(out*".records.csv"),1)
        f=split(line,',');push!(existing,(parse(Int,f[1]),f[2],f[3],f[4],parse(Int,f[5])))
    end
    for line in Iterators.drop(eachline(out*".timings.csv"),1)
        f=split(line,',')
        f[3]=="weights" && push!(existing_weights,(parse(Int,f[1]),f[2],parse(Int,f[5])))
    end
    for line in Iterators.drop(eachline(out*".witness.csv"),1)
        f=split(line,',');push!(existing_witness,(parse(Int,f[1]),f[2],f[3]))
    end
end
const PS=H.ProximalSolvers
const PB=H.ProximalBase
function cached_native(c,X,Z,v,xb,yb,scales,w,a,payload)
    p=c.p;T=eltype(Z);beta=zeros(T,p)
    lambda,info=H._slope_calibration(p,c.n,T;lambda=w,calibration=:bh,q=.1,sigma=1,intercept=false,standardize_x=false)
    f=(one(T)/c.n)*H.LeastSquaresLoss(v,Z);g=PB.SLOPE(lambda)
    if payload=="fit"
        r=PS.solve!(beta,f,g,PS.SLOPEAPG(;maxiter=100000,tol=1e-8))
        beta ./= scales;b0=yb-dot(xb,beta)
        return H.SLOPESolution(beta,b0,false,Vector(c.y .- b0 .- X*beta),lambda,
            r.iterations,r.converged,r.kkt_residual,info)
    end
    r=PS.solve_path(f,g,a,PS.SLOPEAPG(;maxiter=100000,tol=1e-8);x=beta,screening=true)
    bs=[b./scales for b in r.solutions];b0=[yb-dot(xb,b) for b in bs]
    H.SLOPEPath(a,lambda,bs,b0,false,c.n,r.converged,r.iterations,r.kkt_residuals,r.working_sizes,r.kkt_repairs,info)
end
function coefficients(z,engine,payload)
    if engine=="native"
        payload=="fit" ? reshape(vcat(z.x,z.intercept),:,1) : vcat(hcat(z.βpath...),z.β0path')
    else
        vcat(hcat((vec(Matrix(b)) for b in z.coefficients)...),zeros(1,length(z.coefficients)))
    end
end
function output_equal(a,b,payload)
    if payload=="fit"
        a.x==b.x && a.intercept==b.intercept && a.residuals==b.residuals && a.lambda==b.lambda &&
            a.iterations==b.iterations && a.converged==b.converged && a.kkt_residual==b.kkt_residual && a.calibration==b.calibration
    else
        all(getfield(a,f)==getfield(b,f) for f in fieldnames(typeof(a)))
    end
end
mode=resume ? "a" : "w"
open(out*".bin",mode) do dst
 open(out*".records.csv",mode) do records
  open(out*".timings.csv",mode) do timings
   open(out*".witness.csv",mode) do witnesses
    if !resume
    println(records,"seed,method,payload,setting,round,ncolumns,reported_converged")
    println(timings,"seed,method,payload,setting,round,seconds,bytes,gctime")
    println(witnesses,"seed,method,payload,complete_output_equal,public_bytes,prepared_bytes,allocation_difference")
    end
    open(input) do src
     while (c=readcase(src))!==nothing
      X=sc==4 ? sparse(c.X) : c.X
      Z,v,xb,yb,scales=H._slope_data(X,c.y,false,false)
      order=circshift(collect(1:3),mod(rep+c.dataset,3))
      for j in order, payload in ("fit","path")
        w=c.W[:,j]
        amax=maximum(cumsum(sort(abs.(X'*c.y/c.n);rev=true))./cumsum(w))
        factors=payload=="fit" ? [1.] : collect(exp.(range(log(amax),log(.2amax);length=20)))
        if engine=="native"
          public=payload=="fit" ? (() -> H.slope(X,c.y,w;tol=1e-8,max_iter=100000)) :
            (() -> H.SLOPEPath(X,c.y,factors;lambda=w,tol=1e-8,max_iter=100000))
          prepared=() -> cached_native(c,X,Z,v,xb,yb,scales,w,factors,payload)
          default=payload=="fit" ? (() -> H.slope(X,c.y,w)) : (() -> H.SLOPEPath(X,c.y,factors;lambda=w))
          calls=[("public",public),("prepared",prepared),("default",default)]
          if payload=="path"
            push!(calls,("unscreened",() -> H.SLOPEPath(X,c.y,factors;lambda=w,tol=1e-8,max_iter=100000,screening=false)))
          end
          if !((c.seed,METHODS[j],payload) in existing_witness)
            za=public();zb=prepared();equal=output_equal(za,zb,payload)
            equal || error("public/prepared complete output mismatch")
            ba=@allocated public();bb=@allocated prepared()
            println(witnesses,join((c.seed,METHODS[j],payload,equal,ba,bb,ba-bb),','));flush(witnesses)
          end
        else
          jf=() -> SJ.slope(X,c.y;α=factors,λ=w,fit_intercept=false,centering=:none,scaling=:none,
              tol=1e-8,max_it=100000,max_clusters=c.p+1,dev_change_tol=0.0,dev_ratio_tol=1.0)
          calls=[("public",jf)]
        end
        calls=circshift(calls,mod(rep+c.dataset,length(calls)))
        for (setting,call) in calls
          prefix=joinpath(out*"-profiles","$(c.dataset)-$(METHODS[j])-$payload-$setting")
          needs_profile=setting in ("public","prepared") && !ispath(prefix*".done")
          missing_rounds=[r for r in 1:3 if !((c.seed,METHODS[j],payload,setting,r) in existing)]
          isempty(missing_rounds) && !needs_profile && continue
          call() # JIT and workload warmup outside timers
          for round in missing_rounds
            GC.gc(); t=@timed call();z=t.value;B=coefficients(z,engine,payload)
            size(B,2)==length(factors) || error("truncated timed result")
            W=w*factors';writerecord(dst,c,W,B)
            ok=engine=="native" ? (payload=="fit" ? z.converged : all(z.converged)) : "unavailable"
            println(records,join((c.seed,METHODS[j],payload,setting,round,size(B,2),ok),','));flush(records)
            println(timings,join((c.seed,METHODS[j],payload,setting,round,t.time,t.bytes,t.gctime),','));flush(timings)
          end
          if needs_profile
            duration=@elapsed call()
            repeats=clamp(ceil(Int,.05/max(duration,1e-6)),1,100)
            Profile.clear();@profile for _ in 1:repeats;call();end
            open(prefix*".cpu.txt","w") do io
                Profile.print(io;format=:flat,sortedby=:count)
            end
            sample_rate=parse(Float64,get(ENV,"HDM_ALLOC_SAMPLE_RATE","0.02"))
            Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=sample_rate call()
            results=Profile.Allocs.fetch()
            # Buffer stack text rather than issuing one NFS write per frame.
            # This observer runs after all three primary timing rounds.
            buffer=IOBuffer()
            println(buffer,"sample_rate=",sample_rate," allocations=",length(results.allocs))
            for allocation in results.allocs
                println(buffer,allocation.size," ",allocation.type)
                for frame in allocation.stacktrace;println(buffer,frame);end
            end
            write(prefix*".alloc.txt",take!(buffer))
            write(prefix*".done","complete\n")
          end
        end
      end
      # Weight-construction microbenchmarks are not included in fit/path times.
      for (method,call) in [("bh",()->H.slope_bh(c.p;n=c.n)),("paper",()->H.slope_gaussian(c.p;n=c.n)),
                           ("julia",()->SJ.regweights(c.p;n=c.n,q=.1,type=:gaussian)/sqrt(c.n))]
        rounds=[r for r in 1:3 if !((c.seed,method,r) in existing_weights)]
        isempty(rounds) && continue
        call()
        for round in rounds
          t=@timed for _ in 1:1000;call();end
          println(timings,join((c.seed,method,"weights","construction",round,t.time/1000,t.bytes/1000,t.gctime/1000),','));flush(timings)
        end
      end
     end
    end
   end
  end
 end
end
provenance(out*".toml",engine,input);write(out*".done","complete\n")
