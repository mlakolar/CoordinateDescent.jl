using HighDimensionalModels,LinearAlgebra,Statistics,Profile,TOML,SHA
const H=HighDimensionalModels
BLAS.set_num_threads(1)
folder,out,variant,fitout=ARGS;mkpath(out)
n,p,q=parse.(Int,split(read(joinpath(folder,"dims.txt"),String)))
readarr(path,dims...)=open(io->reshape(read!(io,Vector{Float64}(undef,prod(dims))),dims...),path)
X,Y=readarr(joinpath(folder,"X.bin"),n,p),readarr(joinpath(folder,"Y.bin"),n,q)
prepared=TOML.parsefile(joinpath(folder,"prepared.toml"))
case,repstr=rsplit(basename(folder),'-';limit=2);rep=parse(Int,repstr)
configs=[(:cd,false),(:fista,false)]
case=="medium" && append!(configs,[(:cd,true),(:fista,true)])
tag(c)=string(c[1],'-',c[2] ? "adaptive" : "unit")
levels(c)=exp.(range(log(.5),log(.02);length=8))*prepared[c[2] ? "lambda_adaptive" : "lambda_unit"]
fit(c)=LowRankPath(X,Y;method=:sparse,rank=3,λpath=levels(c),block_solver=c[1],adaptive=c[2],
  max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
rss()=parse(Int,match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String)).captures[1])*1024
for c in configs;fit(c);end
metadata=Dict("case"=>case,"rep"=>rep,"variant"=>variant,"cpu"=>Sys.cpu_info()[1].model,
  "julia"=>string(VERSION),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
  "blas"=>string(BLAS.get_config()),"warmed_current_rss"=>rss(),"warmed_peak_rss"=>Sys.maxrss(),
  "source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"lowrank.jl")))))
saved=Dict{String,Any}()
open(joinpath(out,"timings.csv"),"w") do io
  println(io,"variant,case,rep,config,round,seconds,bytes,iterations,converged,B_sha256")
  for rd in 1:3,c in circshift(configs,mod(rd+rep,length(configs)))
    GC.gc();t=@timed fit(c);z=t.value;key=tag(c)
    digest=bytes2hex(sha256(vcat([reinterpret(UInt8,vec(s.B)) for s in z.solutions]...)))
    println(io,join((variant,case,rep,key,rd,t.time,t.bytes,sum(s.iterations for s in z.solutions),all(s.converged for s in z.solutions),digest),','));flush(io)
    if rd==1
      saved[key]=z
      for (j,s) in enumerate(z.solutions)
        label="$key-$j"
        for (k,a) in (("B",s.B),("E",s.residuals),("intercept",s.intercept));open(f->write(f,a),joinpath(out,"$label-$k.bin"),"w");end
        open(f->TOML.print(f,Dict("lambda"=>z.λpath[j],"rank"=>s.rank,"df"=>s.df,"support"=>s.activerows,
          "objective"=>s.objective,"iterations"=>s.iterations,"converged"=>s.converged)),joinpath(out,"$label.toml"),"w")
      end
    end
  end
end
metadata["current_after_timing"]=rss();metadata["peak_after_timing"]=Sys.maxrss()
if case in ("wide","responses","ill")
  for c in configs
    calls=0;began=time();Profile.clear()
    Profile.@profile while calls==0 || (calls<3 && time()-began<.25)
      fit(c);calls+=1
    end
    open(joinpath(out,"cpu-$(tag(c)).txt"),"w") do io
      println(io,"calls=",calls," raw_stack_words=",length(Profile.fetch()));Profile.print(io;format=:flat,sortedby=:count)
    end
  end
end
metadata["whole_worker_peak_before_components"]=Sys.maxrss()
# Independent cached component timings: useful cost probes, not an additive
# decomposition of the public fit. All measured methods remain uninstrumented.
d=H._lr_data(X,Y,true);f=H._lr_svd(d);B0=H._lr_rrcoef(d,f,3);M=f.U'd.Y
XX,XY=d.X'd.X/n,d.X'd.Y/n
S0=svd(B0;full=false);V0=S0.V[:,1:3]
open(joinpath(out,"components.csv"),"w") do io
  println(io,"variant,case,rep,solver,component,round,seconds,bytes")
  for solver in (:cd,:fista)
    key=string(solver,"-0.1-unit")
    A=readarr(joinpath(fitout,"$key-A.bin"),p,3);V=readarr(joinpath(fitout,"$key-V.bin"),q,3)
    B=A*V';w=ones(p);λ=.1*prepared["lambda_unit"];L=prepared["L"];C=XY*V
    function diagnostic()
      E=variant=="candidate" && 3*(p+q)<p*q ? d.Y-(d.X*A)*V' : d.Y-d.X*B
      objective=sum(abs2,E)/(2n)+λ*sum(norm(view(A,j,:)) for j in 1:p)
      if variant=="candidate"
        rows=H._activerows(B);rk=H._lr_factor_rank(B,A,V');k=isempty(rows) ? 0 : length(rows)==p ? f.diagnostic_xrank : rank(d.X[:,rows])
        H._lr_solution(d,B,λ,0,true,objective,:sparse;rows,coefficient_rank=rk,residual=E,df=rk*(k+q-rk)+q)
      else
        H._lr_solution(d,B,λ,0,true,objective,:sparse)
      end
    end
    components=[("centering",()->H._lr_data(X,Y,true)),("design_svd",()->svd(d.X;full=false)),
      ("response_projection",()->f.U'd.Y),("response_svd",()->svd(M;full=false)),
      ("rrr_reconstruction",()->H._lr_rrcoef(d,f,3)),("coefficient_svd",()->svd(B0;full=false)),
      ("initial_A",()->B0*V0),("gram",()->d.X'd.X/n),("cross_product",()->d.X'd.Y/n),
      ("gram_eigen_scale",()->eigmax(Symmetric(XX))),
      ("predictor_update",()->solver==:cd ? H._lr_group!(copy(A),XX,C,λ,w;max_iter=10000,tol=1e-9) : H._lr_group_fista!(copy(A),XX,C,λ,w;max_iter=10000,tol=1e-9,L)),
      ("procrustes",()->(S=svd(XY'A;full=false);S.U*S.Vt)),
      ("coefficient_product",()->A*V'),("stationarity_check",()->H._lr_group_kkt(A,XX*A-C,λ,w)),("result_diagnostics",diagnostic)]
    for (_,fn) in components;fn();end
    for rd in 1:3,(component,fn) in circshift(components,mod(rd+rep,length(components)))
      GC.gc();t=@timed fn();println(io,join((variant,case,rep,solver,component,rd,t.time,t.bytes),','));flush(io)
    end
  end
end
metadata["whole_worker_peak_before_replay"]=Sys.maxrss()
include("probe.jl");install_probe!()
for c in configs
  empty!(HDM102_STATES);foreach(k->HDM102_COUNTS[k]=0,keys(HDM102_COUNTS))
  z=Base.invokelatest(fit,c);key=tag(c);reference=saved[key]
  length(HDM102_STATES)==length(z.solutions)==8 || error("missing actual path factors")
  for (j,(s,ref,state)) in enumerate(zip(z.solutions,reference.solutions,HDM102_STATES))
    norm(s.B-ref.B)<=1e-12*max(1,norm(ref.B)) && s.iterations==ref.iterations || error("path replay differs")
    label="$key-$j"
    for (k,a) in (("A",state.A),("V",state.V),("weights",state.w));open(f->write(f,a),joinpath(out,"$label-$k.bin"),"w");end
  end
  open(f->TOML.print(f,Dict("cd_sweeps"=>HDM102_COUNTS[:cd_sweeps],"fista_steps"=>HDM102_COUNTS[:fista_steps],
    "kkt_calls"=>HDM102_COUNTS[:kkt_calls])),joinpath(out,"$key-probe.toml"),"w")
end
open(io->TOML.print(io,metadata),joinpath(out,"metadata.toml"),"w")
