# Standalone fit-process RSS. Certificates run only after this process exits.
using LinearAlgebra, SparseArrays, Statistics, DelimitedFiles
BLAS.set_num_threads(1)
const ROOT = ENV["RUN"]
const OUT = joinpath(ROOT,"memory")
engine, rep, id = ARGS[1],parse(Int,ARGS[2]),parse(Int,ARGS[3])
if engine == "native"
  if get(ENV,"BENCHMARK_MODULE","HighDimensionalModels") == "CoordinateDescent"
    @eval import CoordinateDescent as Native
  else
    @eval import HighDimensionalModels as Native
  end
elseif engine == "baseline"
  @eval module Baseline
    using LinearAlgebra, Statistics, ProximalBase, ProximalSolvers
    using SpecialFunctions: erfcinv
    import StatsAPI
    include(joinpath(ENV["FROZEN_RUN"],"source","benchmark","selection","sparse","baseline_slope.jl"))
  end
elseif engine == "SLOPE.jl"
  @eval import SLOPE as SJ
else
  error("unknown engine")
end
function loadcase(id)
  open(joinpath(ROOT,"inputs","case-$id.bin")) do io
    n,p,sp,nz,L,intercept,standardize_x=Int.(read!(io,zeros(Int64,7)))
    if sp==1
      ptr=Int.(read!(io,zeros(Int64,p+1)));rows=Int.(read!(io,zeros(Int64,nz)))
      X=SparseMatrixCSC(n,p,ptr,rows,read!(io,zeros(nz)))
    else
      X=read!(io,zeros(n,p))
    end
    y=read!(io,zeros(n));lambda=read!(io,zeros(p));factors=read!(io,zeros(L))
    (;X,y,lambda,factors,intercept=intercept==1,standardize_x=standardize_x==1)
  end
end
function fit(c)
  if engine == "SLOPE.jl"
    f=SJ.slope(c.X,c.y;α=c.factors,λ=c.lambda,fit_intercept=c.intercept,
      centering=c.intercept ? :mean : :none,scaling=c.standardize_x ? :sd : :none,
      tol=1e-9,max_it=100000,max_clusters=size(c.X,2)+1,dev_change_tol=0.0,dev_ratio_tol=1.0)
    length(f.coefficients)==length(c.factors) || error("truncated path")
    hcat((vec(Matrix(b)) for b in f.coefficients)...),c.intercept ? [only(b) for b in f.intercepts] : zeros(length(c.factors))
  else
    backend=engine=="native" ? Native : Baseline
    f=backend.SLOPEPath(c.X,c.y,c.factors;lambda=c.lambda,intercept=c.intercept,
      standardize_x=c.standardize_x,tol=1e-7,max_iter=100000)
    all(f.converged) || error("$engine convergence")
    hcat(f.βpath...),f.β0path
  end
end
c=loadcase(id)
fit(c);fit(c);GC.gc()
B,b0=fit(c)
writedlm(joinpath(OUT,"coef-$rep-$id-$engine.csv"),vcat(transpose(b0),B),',')
println("FIT_ONLY_DONE engine=$engine rep=$rep case=$id")
