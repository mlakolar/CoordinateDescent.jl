# Discovery only. See README.md for resource layout and the exact recipe.
using CoordinateDescent, ProximalBase, Random, LinearAlgebra, Statistics, SparseArrays, Profile, DelimitedFiles
BLAS.set_num_threads(1)
mode = ARGS[1]
out = get(ENV, "CD77_OUTPUT", "outputs")
mkpath(out)
opt = CDOptions(; opt_tol = 1e-8, max_iter = 10000)

function ardesign(rng, n, p, rho)
  X = randn(rng, n, p)
  for j in 2:p
    X[:, j] .= rho .* X[:, j-1] .+ sqrt(1-rho^2) .* X[:, j]
  end
  X
end

if mode == "bench"
  p, rep = parse.(Int, ARGS[2:3])
  n = 200
  X = ardesign(Xoshiro(770000 + rep), n, p, 0.3)
  lambda = 0.75sqrt(2log(p)/n)
  # Compile and warm up on a small problem; no reported warm-up measurement.
  nodewise_lasso(X[:, 1:20], lambda; options = opt)
  for gram in (p == 1000 ? (true, false) : (true,))
    tag = "p$(p)-rep$(rep)-gram$(gram)"
    nodewise_lasso(X, lambda; gram, options = opt, rng = Xoshiro(rep))
    GC.gc()
    timed = @timed nodewise_lasso(X, lambda; gram, options = opt, rng = Xoshiro(rep))
    fit = timed.value
    @assert all(fit.converged)
    @assert maximum(abs(sum(X[:, j] .* fit.scores[:, a])/n - 1) for (a,j) in enumerate(fit.targets)) < 1e-6
    Profile.clear()
    Profile.init(; n = 10^7, delay = 0.001)
    @profile profiled = nodewise_lasso(X, lambda; gram, options = opt, rng = Xoshiro(rep))
    @assert fit.theta ≈ profiled.theta
    open(joinpath(out, "$tag-profile.txt"), "w") do io
      Profile.print(io; format = :flat, sortedby = :count, mincount = 2)
    end
    open(joinpath(out, "$tag.csv"), "w") do io
      println(io, "n,p,rep,gram,threads,lambda,seconds,bytes,gctime,nonzeros,passes,converged")
      println(io, join((n,p,rep,gram,Threads.nthreads(),lambda,timed.time,timed.bytes,timed.gctime,
                       nnz(fit.theta),sum(fit.iterations),all(fit.converged)), ','))
    end
    println("BENCH_COMPLETE ", tag); flush(stdout)
  end
elseif mode == "coverage"
  id = parse(Int, ARGS[2])
  configs = [(cov,s,placement,c) for cov in (:toeplitz,:equicorrelated) for s in (3,15)
             for placement in (:first,:random) for c in (2,4)]
  cov,s,placement,c = configs[id]
  n,p = 100,500
  rng = Xoshiro(cov == :toeplitz ? 771 : 772)
  X = cov == :toeplitz ? ardesign(rng,n,p,0.9) : sqrt(0.8).*randn(rng,n,1) .+ sqrt(0.2).*randn(rng,n,p)
  support = placement == :first ? collect(1:s) : sort(randperm(Xoshiro(773+s),p)[1:s])
  beta = zeros(p); beta[support] = c .* rand(Xoshiro(774+s),s)
  inactive = setdiff(1:p,support)
  proj = nodewise_lasso(X,0.1; options=opt,threads=false,rng=Xoshiro(775))
  @assert all(proj.converged)
  open(joinpath(out,"coverage-$id.csv"),"w") do io
    println(io,"design,s,placement,c,rep,active_coverage,null_coverage,active_length,null_length,simultaneous_coverage,fwer,converged,sigma")
    for rep in 1:100
      y = X*beta + randn(Xoshiro(780000+1000id+rep),n)
      fit = debiased_lasso(X,y;projection=proj,options=opt,rng=Xoshiro(rep))
      @assert fit.converged
      ci = confint(fit)
      covered = (ci[:,1] .<= beta) .& (beta .<= ci[:,2])
      lengths = ci[:,2]-ci[:,1]
      simcov,fwer = NaN,NaN
      if rep <= 20
        boot = multiplier_bootstrap(fit;B=499,rng=Xoshiro(790000+rep))
        simcov = all((boot.intervals[:,1] .<= beta) .& (beta .<= boot.intervals[:,2]))
        fwer = any(boot.adjusted_pvalues[inactive] .<= 0.05)
      end
      println(io,join((cov,s,placement,c,rep,mean(covered[support]),mean(covered[inactive]),
                       mean(lengths[support]),mean(lengths[inactive]),simcov,fwer,fit.converged,fit.sigma),','))
      flush(io)
    end
  end
  println("COVERAGE_COMPLETE ",id)
elseif mode == "glmcoverage"
  id = parse(Int, ARGS[2])
  configs = [(s,placement,c) for s in (3,15) for placement in (:first,:random) for c in (1,2,4)]
  s,placement,c = configs[id]
  n,p = 100,500
  X = ardesign(Xoshiro(771),n,p,0.9)
  support = placement == :first ? collect(1:s) : sort(randperm(Xoshiro(773+s),p)[1:s])
  beta = zeros(p); beta[support] = c .* rand(Xoshiro(774+s),s)
  inactive = setdiff(1:p,support)
  eta = X*beta
  mu = map(t -> t >= 0 ? inv(1+exp(-t)) : exp(t)/(1+exp(t)),eta)
  settings = CDOptions(;opt_tol=1e-7,max_iter=3000)
  open(joinpath(out,"glmcoverage-$id.csv"),"w") do io
    println(io,"design,s,placement,c,rep,active_coverage,null_coverage,active_length,null_length,fit_converged,score_converged,null_score_pvalue,error")
    for rep in 1:20
      y = Float64.(rand(Xoshiro(800000+1000id+rep),n) .< mu)
      try
        initial = glm_lasso(X,y,0.05,ProximalBase.Logistic(),settings;newton_max_iter=50,rng=Xoshiro(rep))
        fit = debiased_lasso(X,y,0.05;family=:binomial,initial=Vector(initial.x),
                             initial_intercept=initial.intercept,nodewise_lambda=0.1,options=settings,rng=Xoshiro(rep))
        valid = initial.converged && fit.converged
        ci = confint(fit)
        covered = (ci[:,1] .<= beta) .& (beta .<= ci[:,2])
        lengths = ci[:,2]-ci[:,1]
        score = decorrelated_score(X,y,first(inactive),0.05;family=:binomial,
                                    projection_lambda=0.03,options=settings,rng=Xoshiro(rep))
        values = valid ? (mean(covered[support]),mean(covered[inactive]),mean(lengths[support]),mean(lengths[inactive])) : (NaN,NaN,NaN,NaN)
        println(io,join((:toeplitz,s,placement,c,rep,values...,valid,score.converged,
                         score.converged ? score.pvalue : NaN,""),','))
      catch err
        message = replace(sprint(showerror,err),','=>';','\n'=>' ')
        println(io,join((:toeplitz,s,placement,c,rep,NaN,NaN,NaN,NaN,false,false,NaN,message),','))
      end
      flush(io)
    end
  end
  println("GLM_COVERAGE_COMPLETE ",id)
else
  error("mode must be bench, coverage or glmcoverage")
end
