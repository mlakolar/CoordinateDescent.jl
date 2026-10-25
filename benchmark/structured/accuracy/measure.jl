using HighDimensionalModels, LinearAlgebra, SparseArrays, DelimitedFiles, Statistics
BLAS.set_num_threads(1)
const HDM = HighDimensionalModels
seed = parse(Int,ARGS[1])
input, reference, out, setting = ARGS[2:5]
mkpath(out)
# Independent per-level diagnostics accompany each saved accuracy check.
fitlog=open(joinpath(out,"fits-$seed-$setting.tsv"),"w")
println(fitlog,"seed\tcase\tsetting\tlevel\tlambda\tcoefficient_error\tobjective_error\tprimal_residual\tdual_residual\titerations\tconverged")
tab,header = readdlm(joinpath(input,"cases.tsv"),'\t',Any;header=true)
names = Symbol.(vec(header))
function case_row(row)
  values = map((n,v)->n in (:n,:p,:order,:path) ? Int(v) : n in (:lo,:hi) ? Float64(v) : v,names,row)
  NamedTuple{Tuple(names)}(Tuple(values))
end
read_input(name) = readdlm(joinpath(input,string(seed),name),'\t',Float64)
function load_case(c)
  if c.family == "signal"
    y = vec(read_input("signal-$(c.n)-y.tsv"))
    x = c.grid == "unit" ? nothing : vec(read_input("irregular-x.tsv"))
    X = nothing; D = difference_matrix(c.p,c.order+1;x)
  else
    X = read_input("reg-$(c.p)-X.tsv"); y = vec(read_input("reg-$(c.p)-y.tsv")); x = nothing
    D = c.family == "fused" ? [.25spdiagm(0=>ones(c.p));.75difference_matrix(c.p)] :
      graph_difference(c.p,[(j,j+1) for j in 1:c.p-1] ∪ [(j,c.p-j+1) for j in 1:c.p÷5])
  end
  levels = (c.path>0 ? exp.(range(log(c.hi),log(c.lo);length=c.path)) : [c.lo])./c.n
  (;X,y,x,D,levels)
end
function solve(c,a)
  if c.family == "signal"
    c.path>0 ? trend_filter_path(a.y,a.levels;order=c.order,x=a.x) :
      trend_filter(a.y,only(a.levels);order=c.order,x=a.x)
  elseif c.family == "fused"
    c.path>0 ? fused_lasso_path(a.X,a.y,a.levels;l1_ratio=.25) :
      fused_lasso(a.X,a.y,.25only(a.levels),.75only(a.levels))
  else
    c.path>0 ? GeneralizedLassoPath(a.X,a.y,a.D,a.levels) :
      generalized_lasso(a.X,a.y,a.D,only(a.levels))
  end
end
fits(s) = s isa GeneralizedLassoPath ? s.solutions : [s]
function objective(B,a)
  residual = (a.X === nothing ? B : a.X*B) .- a.y
  vec(sum(abs2,residual;dims=1))./(2length(a.y)) .+ a.levels.*vec(sum(abs,a.D*B;dims=1))
end
# Warm actual argument/record types on unrelated small inputs.
for row in eachrow(tab)
  c=case_row(row); p=12; n=c.family=="signal" ? p : 32
  cw=merge(c,(n=n,p=p))
  x=c.grid=="unit" ? nothing : cumsum(fill(1.1,p))
  X=c.family=="signal" ? nothing : reshape(sin.(1:n*p),n,p)+Matrix{Float64}(I,n,p)
  D=c.family=="graph" ? graph_difference(p,[(j,j+1) for j in 1:p-1]) : difference_matrix(p)
  a=(;X,y=cos.(1:n),x,D,levels=c.path>0 ? [.1,.01] : [.01])
  solve(cw,a)
end
open(joinpath(out,"timings-$seed-$setting.tsv"),"w") do io
  println(io,"seed\tcase\tsetting\tround\tseconds\tbytes\tconverged\titerations\tstatus")
  open(joinpath(out,"accuracy-$seed-$setting.tsv"),"w") do ai
    println(ai,"seed\tcase\tsetting\tcoefficient_error\tobjective_error\tfinite\tcolumns\tconverged\tpass\tdual_gap")
    for row in eachrow(tab)
      c=case_row(row); a=load_case(c); q=max(c.path,1)
      ref=reshape(read!(joinpath(reference,"reference-$(c.id).bin"),Vector{Float64}(undef,c.p*q)),c.p,q)
      for round in 1:3
        GC.gc(); m=@timed solve(c,a); fs=fits(m.value)
        println(io,join((seed,c.id,setting,round,m.time,m.bytes,all(s.converged for s in fs),sum(s.iterations for s in fs),"ok"),'\t'));flush(io)
        if round==1
          B=reduce(hcat,getproperty.(fs,:x)); ob,oref=objective(B,a),objective(ref,a)
          ce=maximum(norm(B[:,j]-ref[:,j])/max(norm(ref[:,j]),sqrt(c.p)*eps()) for j in 1:q)
          oe=maximum(abs.(ob.-oref)./max.(abs.(oref),eps()))
          dg=NaN
          if c.family!="signal"
            u=read!(joinpath(reference,"dual-$(c.id).bin"),Vector{Float64}(undef,size(a.D,1)))
            # Stable primal-dual gap for the unnormalized squared loss.
            H=a.X'a.X; h=a.X'a.y; alpha=c.n*last(a.levels)
            @assert maximum(abs,u) <= alpha
            bd=H\(h-a.D'u); delta=B[:,end]-bd
            gap=dot(delta,H*delta)/2+alpha*sum(abs,a.D*B[:,end])-dot(u,a.D*B[:,end])
            dg=gap/max(c.n*last(ob),eps())
          end
          for j in 1:q
            fj=fs[j]
            cj=norm(B[:,j]-ref[:,j])/max(norm(ref[:,j]),sqrt(c.p)*eps())
            oj=abs(ob[j]-oref[j])/max(abs(oref[j]),eps())
            println(fitlog,join((seed,c.id,setting,j,a.levels[j],cj,oj,fj.primal_residual,fj.dual_residual,fj.iterations,fj.converged),'\t'));flush(fitlog)
          end
          finite=all(isfinite,B)&&all(isfinite,ref)
          pass=finite&&ce<=1e-4&&oe<=1e-6&&all(s.converged for s in fs)
          println(ai,join((seed,c.id,setting,ce,oe,finite,q,all(s.converged for s in fs),pass,dg),'\t'));flush(ai)
          # Retain only compact tables locally; matrices remain in this run.
          open(joinpath(out,"beta-$seed-$setting-$(c.id).bin"),"w") do bio;write(bio,B);end
        end
      end
    end
  end
end

close(fitlog)
