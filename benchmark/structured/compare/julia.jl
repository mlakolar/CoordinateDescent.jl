using CoordinateDescent, LinearAlgebra, SparseArrays, DelimitedFiles, Statistics, Profile
BLAS.set_num_threads(1)
seed, input, out, mode = parse(Int,ARGS[1]), ARGS[2], ARGS[3], ARGS[4]
mkpath(out)
table, header = readdlm(joinpath(input,"cases.tsv"),'\t',Any;header=true)
names = Symbol.(vec(header))
const CD = CoordinateDescent
function case_row(row)
  vals = map((n,v)->n in (:n,:p,:order,:path) ? Int(v) : n in (:lo,:hi) ? Float64(v) : v,names,row)
  NamedTuple{Tuple(names)}(Tuple(vals))
end
readinput(name) = readdlm(joinpath(input,string(seed),name),'\t',Float64)
function loadcase(c)
  if c.family == "signal"
    y = vec(readinput("signal-$(c.n)-y.tsv"))
    x = c.grid == "unit" ? nothing : vec(readinput("irregular-x.tsv"))
    X = nothing
    D = difference_matrix(c.p,c.order+1;x)
  else
    X = readinput("reg-$(c.p)-X.tsv"); y = vec(readinput("reg-$(c.p)-y.tsv")); x = nothing
    DD = difference_matrix(c.p)
    D = c.family == "fused" ? [.25spdiagm(0=>ones(c.p)); .75DD] :
        c.family == "trend" ? difference_matrix(c.p,2) :
        graph_difference(c.p,[(j,j+1) for j in 1:c.p-1] ∪ [(j,c.p-j+1) for j in 1:c.p÷5])
  end
  levels = c.path>0 ? exp.(range(log(c.hi),log(c.lo);length=c.path)) : [c.lo]
  (; X,y,x,D,levels=levels ./ c.n)
end
function solve(c,a; tight=false)
  matched = hasproperty(c,:julia_setting) ? c.julia_setting=="optimized-matched" :
    get(ENV,"COMPARE_TIGHT","false")=="true"
  # The initial cyclic-graph comparison showed that ADMM residual tolerances
  # need an extra decade on the correlated design to meet coefficient accuracy.
  strict = tight || (matched && c.family=="graph")
  opts = (;abstol=strict ? 1e-9 : matched ? 1e-8 : 1e-7,
    reltol=strict ? 1e-8 : matched ? 1e-7 : 1e-6,max_iter=20000)
  if c.family=="signal"
    c.path>0 ? trend_filter_path(a.y,a.levels;order=c.order,x=a.x,opts...) :
      trend_filter(a.y,only(a.levels);order=c.order,x=a.x,opts...)
  elseif c.family=="fused"
    c.path>0 ? fused_lasso_path(a.X,a.y,a.levels;l1_ratio=.25,opts...) :
      fused_lasso(a.X,a.y,.25only(a.levels),.75only(a.levels);opts...)
  elseif c.family=="trend"
    c.path>0 ? trend_filter_path(a.X,a.y,a.levels;order=1,opts...) :
      trend_filter(a.X,a.y,only(a.levels);order=1,opts...)
  else
    c.path>0 ? GeneralizedLassoPath(a.X,a.y,a.D,a.levels;opts...) :
      generalized_lasso(a.X,a.y,a.D,only(a.levels);opts...)
  end
end
fits(s) = s isa GeneralizedLassoPath ? s.solutions : [s]
function rss()
  line=only(filter(s->startswith(s,"VmHWM:"),readlines("/proc/self/status")))
  parse(Int,split(line)[2])*1024
end
# Compile the actual case-record and input types on unrelated small problems,
# including irregular grids and signal paths, before measuring any dataset.
warmed=Set{Any}()
for row in eachrow(table)
  c=case_row(row); key=(c.family,c.order,c.grid,c.path>0)
  key in warmed && continue
  push!(warmed,key)
  signal=c.family=="signal"; p=signal ? 32 : 8
  cw=merge(c,(n=32,p=p))
  x=c.grid=="unit" ? nothing : cumsum(fill(1.1,p))
  X=signal ? nothing : reshape(sin.(1:256),32,8)+Matrix{Float64}(I,32,8)
  a=(;X,y=cos.(1:32),x,D=difference_matrix(p),levels=c.path>0 ? [.1,.01] : [.01])
  solve(cw,a)
end
open(joinpath(out,"Julia-$seed-$mode.tsv"),"w") do io
  println(io,"seed\tcase\tsolver\tround\tseconds\tbytes\tconverged\titerations\tobjective\tworker_hwm_bytes\tstatus")
  for row in eachrow(table)
    c=case_row(row)
    canary = c.id in ("signal-n1000-k2-l1","signal-n10000-k2-l10","graph-path-p40")
    mode=="canary" && !canary && continue
    mode=="full" && canary && continue
    mode=="profile" && !(c.id in ("signal-n10000-k2-l1","graph-path-p100","fused-path-p100")) && continue
    mode=="gram" && !(c.path>0 && (c.family=="fused" || (c.family=="signal" && c.order==0))) && continue
    mode=="gram-profile" && c.id!="fused-path-p100" && continue
    a=loadcase(c)
    if mode in ("profile","gram-profile")
      solve(c,a); GC.gc(); Profile.clear()
      Profile.@profile for _ in 1:(mode=="gram-profile" ? 10 : 1); solve(c,a); end
      open(joinpath(out,"profile-$seed-$(c.id).txt"),"w") do pio
        Profile.print(pio;format=:flat,sortedby=:count,mincount=3)
      end
      Profile.Allocs.clear()
      Profile.Allocs.@profile sample_rate=0.01 solve(c,a)
      allocs=Profile.Allocs.fetch().allocs
      open(joinpath(out,"alloc-$seed-$(c.id).txt"),"w") do pio
        for al in sort(allocs;by=a->a.size,rev=true)[1:min(30,length(allocs))]
          println(pio,"bytes=",al.size," type=",al.type)
          foreach(f->println(pio,"  ",f),al.stacktrace)
        end
      end
      continue
    end
    for round in 1:3
      try
        GC.gc(); m=@timed solve(c,a); fs=fits(m.value)
        if round==1
          B=reduce(hcat,getproperty.(fs,:x))
          open(joinpath(out,"julia-$seed-$(c.id).bin"),"w") do bio; write(bio,B); end
        end
        println(io,join((seed,c.id,"julia",round,m.time,m.bytes,all(s->s.converged,fs),
          sum(s.iterations for s in fs),last(fs).objective,rss(),"ok"),'\t')); flush(io)
      catch e
        println(io,join((seed,c.id,"julia",round,NaN,0,false,0,NaN,rss(),replace(sprint(showerror,e),'\n'=>' ')),'\t')); flush(io)
      end
    end
  end
end
