# One independent process per replication; all timing/profiling under Slurm.
using HighDimensionalModels, LinearAlgebra, Random, Statistics, DelimitedFiles, Profile
BLAS.set_num_threads(1)
rep = parse(Int,ARGS[1]); out = ARGS[2]; mkpath(out)
function require_converged(s)
  c = s.converged
  (c isa Bool ? c : all(c)) || error("nonconverged SLOPE fit")
  s
end
function benchmark_workloads(rep)
  rng = Xoshiro(82000) # fixed data across independent process replications
  X = randn(rng,300,120); y = X[:,1:10]*fill(0.4,10)+randn(rng,300)
  H = randn(rng,150,500); H[:,2:end] .+= 0.5 .* H[:,1]
  v = H[:,1:8]*fill(0.5,8)+randn(rng,150)
  S = [0.4^abs(i-j) for i in 1:120,j in 1:120]
  pairs = complementary_pairs(300,20;rng=Xoshiro(5))
  # Each closure owns its data; no mutation of shared solver state.
  ["slope-single" => () -> require_converged(slope(X,y)),
   "slope-path-screened" => () -> require_converged(SLOPEPath(X,y;nλ=20,λmin_ratio=0.15)),
   "slope-path-full" => () -> require_converged(SLOPEPath(X,y;nλ=20,λmin_ratio=0.15,screening=false)),
   "slope-wide-screened" => () -> require_converged(SLOPEPath(H,v;nλ=15,λmin_ratio=0.25)),
   "slope-wide-full" => () -> require_converged(SLOPEPath(H,v;nλ=15,λmin_ratio=0.25,screening=false)),
   "gaussian-construction" => () -> gaussian_knockoffs(X,zeros(120),S;rng=Xoshiro(1)),
   "fixed-filter" => () -> knockoff_filter(X,y,0.08;method=:fixed,rng=Xoshiro(1)),
   "stability-serial" => () -> stability_selection(X,y,0.1;q=10,pairs,parallel=false,rng=Xoshiro(1)),
   "stability-threaded" => () -> stability_selection(X,y,0.1;q=10,pairs,parallel=true,rng=Xoshiro(1))]
end
function measure(rep,out)
  workloads = benchmark_workloads(rep)
  # Rotate order across processes to avoid a fixed warm-up/order effect.
  workloads = circshift(workloads,rep-1)
  open(joinpath(out,"timing-$rep.csv"),"w") do io
    println(io,"rep,case,seconds,bytes,gctime")
    for (name,f) in workloads
      f(); GC.gc()
      r = @timed f()
      println(io,"$rep,$name,$(r.time),$(r.bytes),$(r.gctime)");flush(io)
      Profile.clear()
      # Profile every case in every replication for >= 0.2 s, after warm-up.
      calls = 0; started = time()
      Profile.@profile while calls < 1 || time()-started < 0.2
        f(); calls += 1
      end
      open(joinpath(out,"profile-$rep-$name.txt"),"w") do pi
        println(pi,"rep=$rep case=$name calls=$calls Julia=$VERSION threads=$(Threads.nthreads()) job=$(get(ENV,"SLURM_JOB_ID","none"))")
        Profile.print(pi;format=:flat,sortedby=:count,mincount=1)
      end
    end
  end
end
function montecarlo(rep,out)
  open(joinpath(out,"mc-$rep.csv"),"w") do io
    println(io,"rep,seed,method,design,q,selected,false,fdp,power")
    for b in 1:50
      seed = 820000 + 1000rep + b
      rng = Xoshiro(seed)
      p, n, k, q = 100, 240, 20, 0.1
      # Orthogonal model of Bogdan et al.: X'X/n=I and known sigma=1.
      X = sqrt(n) .* Matrix(qr(randn(rng,n,p)).Q)
      beta = [fill(4.5/sqrt(n),k);zeros(p-k)]
      y = X*beta + randn(rng,n)
      s = require_converged(slope(X,y;q,sigma=1,tol=1e-7))
      record(io,rep,seed,"slope","orthogonal",q,findall(!iszero,s.x),k)
      # Fixed-X linear Gaussian experiment with normalized random columns.
      F = randn(rng,n,p); F ./= sqrt.(sum(abs2,F;dims=1)./n)
      yf = F*beta + randn(rng,n)
      f = knockoff_filter(F,yf,0.04;method=:fixed,q,rng=Xoshiro(seed+1),opt_tol=1e-9)
      record(io,rep,seed,"fixed-knockoff+","gaussian-normalized",q,f.selected,k)
      # Model-X Gaussian AR(1), a design family from Candes et al.
      S = [0.4^abs(i-j) for i in 1:p,j in 1:p]
      G = randn(rng,n,p)*cholesky(S).U
      yg = G*beta + randn(rng,n)
      g = knockoff_filter(G,yg,0.04;mu=zeros(p),Sigma=S,q,rng=Xoshiro(seed+2),opt_tol=1e-9)
      record(io,rep,seed,"model-X-knockoff+","AR1-rho0.4",q,g.selected,k)
      # CPSS controls PFER under its low-selection-probability assumption;
      # FDR is recorded descriptively, without comparing it to q=0.1.
      st = stability_selection(G,yg,0.12;q=10,cutoff=0.8,npairs=20,rng=Xoshiro(seed+3),parallel=false,opt_tol=1e-8)
      record(io,rep,seed,"CPSS","AR1-rho0.4",NaN,st.selected,k)
      flush(io)
    end
  end
end
function record(io,rep,seed,method,design,q,selected,k)
  V = count(>(k),selected); R = length(selected)
  println(io,"$rep,$seed,$method,$design,$q,$R,$V,$(V/max(1,R)),$((R-V)/k)")
end
measure(rep,out)
montecarlo(rep,out)
println("REPLICATION_DONE $rep")
