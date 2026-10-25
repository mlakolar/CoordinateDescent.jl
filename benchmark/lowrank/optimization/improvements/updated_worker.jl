# Reuse the frozen comparison adapters and certificates; this driver measures
# the new public options and the previous public API in separate environments.
ENV["CD81_LIBRARY"] = "1"
include(joinpath(@__DIR__, "julia_worker.jl"))

const AFTER = ["cd_rank", "cd_nuclear", "cd_panoc", "ps_fista", "ps_admm", "pa_fista", "pa_panoc",
  "cd_sparse", "block_fista", "cd_sparse_api", "cd_sparse_fista", "cd_component", "component_fista"]
const BEFORE = ["before_rank", "before_nuclear", "before_sparse"]
updated_family(s) = occursin("rank", s) ? "rank" : occursin("sparse", s) || s == "block_fista" ? "sparse" : occursin("component", s) ? "component" : "nuclear"

function execute(s, d, frac; mode = "fit", cached = nothing)
  if s in ("before_rankpath", "cd_rankpath")
    p = CD.LowRankPath(d.X, d.Y; method = :rank, ranks = 0:3)
    return last(p.solutions).B, 0, true, ""
  elseif s == "before_rank"
    z = CD.reduced_rank(d.X, d.Y, d.r)
    return z.B, z.iterations, z.converged, ""
  elseif s in ("cd_nuclear", "cd_panoc", "before_nuclear")
    which = s == "cd_panoc" ? :panoc : :apg
    opts = s == "before_nuclear" ? (;) : (; solver = which)
    z = if mode == "path"
      CD.LowRankPath(d.X, d.Y; λpath = exp.(range(log(.5), log(.02); length = 8)).*d.scalars[1],
        max_iter = 20000, tol = 1e-8, opts...)
    elseif mode == "kernel"
      CD._lr_nuclear(cached.d, cached.g, frac*d.scalars[1], zeros(d.p, d.q);
        max_iter = 20000, tol = 1e-8, opts...)
    else
      CD.nuclear_regression(d.X, d.Y, frac*d.scalars[1]; max_iter = 20000, tol = 1e-8, opts...)
    end
    return z isa CD.LowRankPath ? (last(z.solutions).B, sum(t.iterations for t in z.solutions), all(t.converged for t in z.solutions), "") :
      (z.B, z.iterations, z.converged, "")
  elseif s in ("cd_sparse_api", "cd_sparse_fista", "before_sparse")
    opts = s == "before_sparse" ? (;) : (; block_solver = s == "cd_sparse_fista" ? :fista : :cd)
    z = CD.sparse_reduced_rank(d.X, d.Y, d.r, frac*d.scalars[2]; weights = ones(d.p),
      max_iter = 1000, tol = 1e-8, inner_max_iter = 10000, inner_tol = 1e-9, opts...)
    return z.B, z.iterations, z.converged, ""
  elseif mode == "kernel"
    return nuclear(s, d, frac; cached)
  end
  runfit(s, d, frac; mode)
end

function api_factors(d, frac, expected; block_solver = :cd, weights = ones(d.p), maximum = d.scalars[2])
  z = CD._lr_data(d.X, d.Y, true)
  initial = CD._lr_rrcoef(z, CD._lr_svd(z), d.r)
  F = svd(initial); V = F.V[:, 1:d.r]; A = initial * V
  H, C = z.X'z.X/d.n, z.X'z.Y/d.n
  L = block_solver == :fista ? eigmax(Symmetric(H)) : 0.0
  B = A*V'; λ = frac*maximum
  for k in 1:1000
    old = B
    inner = block_solver == :fista ?
      CD._lr_group_fista!(A, H, C*V, λ, weights; max_iter = 10000, tol = 1e-9, L) :
      CD._lr_group!(A, H, C*V, λ, weights; max_iter = 10000, tol = 1e-9)
    S = svd(C'A); V = S.U*S.Vt; B = A*V'
    if inner && norm(B-old) <= 1e-8*max(1,norm(B)) &&
        CD._lr_group_kkt(A,H*A-C*V,λ,weights) <= 1e-9*max(1,norm(C*V))
      break
    end
  end
  norm(B-expected) <= 1e-7*max(1,norm(expected)) || error("public sparse factor replay mismatch")
  A, V
end

function measure!(io, d, out, s, frac, rd, phase; mode = "fit")
  cached = mode == "kernel" ? cache(d) : nothing
  FACTORS[] = nothing
  GC.gc(false)
  t = @elapsed B, it, ok, detail = execute(s,d,frac;mode,cached)
  fam = updated_family(s)
  level = mode == "path" ? .02 : frac
  if s in ("cd_sparse_api", "cd_sparse_fista", "before_sparse")
    FACTORS[] = api_factors(d, frac, B; block_solver = s == "cd_sparse_fista" ? :fista : :cd)
  elseif s == "cd_sparse"
    FACTORS[] = sparse_factors(d, frac, B)
  end
  obj,gap,res = certificate(d,fam,B,level;factors=FACTORS[])
  status = res <= 1e-6 && (isnan(gap) || gap <= 1e-6) ? "passed" : "accuracy_failed"
  detail == "timeout" && (status = "timeout")
  println(io, join((phase,"julia",d.case,d.rep,fam,s,mode,level,rd,t,it,obj,gap,res,status,ok,
    Sys.maxrss()/1024,RSS_BASELINE[],detail), ',')); flush(io)
  if rd == 1 && mode == "fit"
    open(f->write(f,B),joinpath(out,"$s-$frac.bin"),"w")
    if fam == "sparse"
      A,V = FACTORS[]
      open(f->write(f,A),joinpath(out,"$s-$frac.A"),"w")
      open(f->write(f,V),joinpath(out,"$s-$frac.V"),"w")
    end
  end
  println(s," ",d.case," rep",d.rep," ",mode," ",level," ",round(t;digits=4),"s ",status);flush(stdout)
end

function updated_main()
  input,out,variant = ARGS[1:3];mkpath(out);d=problem(input)
  memory = length(ARGS)>=4 ? ARGS[4] : ""
  solvers = variant == "before" ? BEFORE : AFTER
  !isempty(memory) && (solvers=[memory])
  warm=problem(joinpath(dirname(input),"small-1"))
  warmwide=merge(warm,(;n=30,X=warm.X[1:30,:],Y=warm.Y[1:30,:],layer=warm.layer[1:30,:]))
  for s in solvers, w in (warm,warmwide)
    execute(s,w,.1)
    if updated_family(s)=="nuclear"
      execute(s,w,.1;mode="kernel",cached=cache(w))
      execute(s,w,.1;mode="path")
    end
  end
  RSS_BASELINE[]=Sys.maxrss()/1024
  println("Julia ",VERSION," BLAS ",BLAS.get_config()," threads=",BLAS.get_num_threads())
  open(joinpath(out,isempty(memory) ? "julia.csv" : "memory.csv"),"w") do io
    println(io,join(FIELDS,','))
    if !isempty(memory)
      measure!(io,d,out,memory,.1,1,variant;mode="memory")
      return
    end
    for rd in 1:3, s in circshift(solvers,mod(d.rep+rd,length(solvers)))
      for frac in (updated_family(s)=="rank" ? (0.0,) : (.5,.1,.02))
        measure!(io,d,out,s,frac,rd,variant)
      end
    end
    nuc=filter(s->updated_family(s)=="nuclear",solvers)
    for s in nuc, mode in (variant=="before" ? ("path",) : ("kernel","path")),rd in 1:3
      measure!(io,d,out,s,.1,rd,variant;mode)
    end
    rp=variant=="before" ? "before_rankpath" : "cd_rankpath"
    for rd in 1:3;measure!(io,d,out,rp,0.0,rd,variant);end
  end
  if d.case in ("medium","responses","wide")
    specs=[(s,"fit") for s in solvers]
    push!(specs,(variant=="before" ? "before_rankpath" : "cd_rankpath","fit"))
    append!(specs,variant=="before" ? [("before_nuclear","path")] : [("cd_nuclear","path"),("cd_panoc","path")])
    for (s,mode) in specs
      Profile.clear();began=time();calls=0
      @profile while time()-began < .25 || calls==0
        execute(s,d,.1;mode);calls+=1
      end
      suffix=mode=="path" ? "-path" : ""
      open(joinpath(out,"profile-$s$suffix.txt"),"w") do f
        println(f,"case=",d.case," rep=",d.rep," calls=",calls," samples=",length(Profile.fetch()))
        Profile.print(f;format=:flat,sortedby=:count,mincount=2)
      end
    end
  end
end
get(ENV,"CD81_IMPROVEMENTS_LIBRARY","0")=="1" || updated_main()
