using LinearAlgebra, SparseArrays, Statistics, Random, DelimitedFiles
import HighDimensionalModels as CD
import ProximalBase as PB
BLAS.set_num_threads(1)
const ROOT=ENV["RUN"]
const OUT=joinpath(ROOT,get(ENV,"RESULTS_DIR","results"))
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
function certificate(c,B,b0)
  # Independent original-input calculation; no production sparse operator.
  X,y=c.X,c.y;n,p=size(X)
  mu=c.intercept ? vec(sum(X;dims=1))./n : zeros(p)
  s=ones(p)
  if c.standardize_x
    if issparse(X)
      for j in 1:p
        r=nzrange(X,j);a=(n-length(r))*mu[j]^2
        for k in r;a+=(nonzeros(X)[k]-mu[j])^2;end
        s[j]=sqrt(a/n)
      end
    else
      s=vec(sqrt.(sum(abs2,X.-transpose(mu);dims=1)./n))
    end
    replace!(s,0.0=>1.0)
  end
  v=y.-(c.intercept ? mean(y) : 0.0)
  g0=(X'*v.-mu.*sum(v))./s./n
  scale=max(1.0,norm(g0,Inf));maxres=0.0;objs=Float64[]
  for k in eachindex(c.factors)
    b=B[:,k];theta=b.*s;r=X*b.+b0[k].-y
    grad=(X'*r.-mu.*sum(r))./s./n
    z=similar(theta);PB.prox!(z,PB.SLOPE(c.factors[k]*c.lambda),theta-grad,1.0)
    maxres=max(maxres,norm(theta-z,Inf)/scale)
    push!(objs,sum(abs2,r)/(2n)+dot(c.factors[k]*c.lambda,sort(abs.(theta);rev=true)))
    c.intercept && (maxres=max(maxres,abs(mean(r))/scale))
  end
  maxres,objs
end
