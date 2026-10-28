# Row selection on a fixed signal grid. Only the training responses enter the
# solver; the observation Gram is diagonal, never a materialized identity.
struct _MaskedSignalDesign{T} <: AbstractMatrix{T}
  p::Int
  observed::Vector{Int}
end
Base.size(A::_MaskedSignalDesign)=(length(A.observed),A.p)
Base.getindex(A::_MaskedSignalDesign{T},i::Int,j::Int) where {T}=T(A.observed[i]==j)
Base.:*(A::_MaskedSignalDesign,b::AbstractVector)=b[A.observed]
function LinearAlgebra.mul!(out::AbstractVector,A::_MaskedSignalDesign,b::AbstractVector)
  for i in eachindex(A.observed);out[i]=b[A.observed[i]];end
  out
end
function LinearAlgebra.mul!(out::AbstractVector,At::Transpose{T,<:_MaskedSignalDesign},b::AbstractVector) where {T}
  A=parent(At);fill!(out,zero(eltype(out)))
  for i in eachindex(A.observed);out[A.observed[i]]=b[i];end
  out
end
_structured_product(A::_MaskedSignalDesign,b)=b[A.observed]
function _structured_tproduct(A::_MaskedSignalDesign{T},b) where {T}
  out=zeros(T,A.p);mul!(out,transpose(A),b)
end
function _structured_gram(A::_MaskedSignalDesign{T}) where {T}
  mask=zeros(T,A.p);mask[A.observed].=one(T);Diagonal(mask)
end
_structured_curvature(A::_MaskedSignalDesign)=size(A,1)/size(A,2)
function _structured_data(A::_MaskedSignalDesign{T},b,intercept) where {T}
  intercept && throw(ArgumentError("a signal fit already includes its constant component; use intercept = false."))
  length(b)==size(A,1) || throw(DimensionMismatch("training responses do not match mask."))
  all(isfinite,b) || throw(ArgumentError("training responses must be finite."))
  A,Vector{T}(b),zeros(T,A.p),zero(T)
end

# The span within a sparse row is the bandwidth of K'K. Unsupported widths
# retain the sparse solver and its full-column-rank failure, without a ridge.
function _masked_bandwidth(K)
  R=sparse(transpose(K));width=0
  for j in axes(R,2)
    rows=nzrange(R,j)
    isempty(rows) || (width=max(width,maximum(rowvals(R)[rows])-minimum(rowvals(R)[rows])))
    width>3 && return nothing
  end
  width
end

function _signal_admm_errors(A,b,D,K,levels,emit;family=:generalized,trend_order=nothing,
    spacing=1,intercept::Bool=false,rho=nothing,max_iter::Integer=20_000,
    abstol::Real=_structured_tolerances(A,b,family,trend_order).abstol,
    reltol::Real=_structured_tolerances(A,b,family,trend_order).reltol,df_tol::Real=1e-4)
  intercept && throw(ArgumentError("signal cross-validation requires intercept = false."))
  _structured_options(max_iter,abstol,reltol,df_tol)
  rho===nothing || (isfinite(rho) && rho>0) || throw(ArgumentError("rho must be positive and finite."))
  T=eltype(b);n,p=size(A);gram=_structured_gram(A);xy=_structured_tproduct(A,b)
  bandwidth=_masked_bandwidth(K);curvature=n/p
  state,factor,last_rho=nothing,nothing,nothing
  for (j,λ) in enumerate(levels)
    if λ==0 || size(D,1)==0
      # The explicit observation-selection design's minimum-norm LS solution.
      emit(copy(xy),j,true)
      continue
    end
    ρ=rho===nothing ? (trend_order===nothing ? _generalized_rho(curvature,n,λ,T) :
      _trend_rho(A,curvature,n,λ,spacing,trend_order,T)) : T(rho)
    if last_rho!=ρ
      factor=_split_factor(gram,K,ρ;bandwidth);last_rho=ρ
    end
    g=trend_order===nothing ? NormL1(T(n*λ)) : TotalVariation1D(T(n*λ))
    β,it,ok,pr,dr,state=_split_fit(A,b,K,g,ρ;max_iter,abstol,reltol,start=state,gram,xy,factor)
    emit(β,j,ok)
  end
  nothing
end

function _signal_fused_errors(A,b,levels,emit;l1_ratio=.5,kwargs...)
  data=_structured_data(A,b,get(kwargs,:intercept,false));T=eltype(b);n=length(b)
  gram=_structured_gram(A)/n;xy=_structured_tproduct(A,b)/n
  start=nothing
  for (j,λ) in enumerate(levels)
    β,it,ok,dr=fused_lasso(A,b,l1_ratio*λ,(1-l1_ratio)*λ;
      _start=start,_data=data,_gram=gram,_xy=xy,_prediction_only=true,kwargs...)
    emit(β,j,ok);start=β
  end
  nothing
end

function _signal_cv(makepath,foldpath,y,levels;nfolds::Integer=10,folds=nothing,
    rng::AbstractRNG=default_rng(),threads::Bool=true,max_concurrent_folds::Integer=2)
  n=length(y)
  n>=2 || throw(ArgumentError("cross-validation needs at least two observations."))
  max_concurrent_folds>0 || throw(ArgumentError("max_concurrent_folds must be positive."))
  fl,K=_foldlabels(rng,n,nfolds,folds)
  all(k->1<=k<=K,fl) || throw(ArgumentError("fold labels must be positive."))
  path=makepath(levels)
  all(s->s.converged,path.solutions) || error("the full-data path did not converge.")
  T=eltype(path.λpath);errs=zeros(T,K,length(path.λpath))
  function fold!(k)
    train,test=findall(!=(k),fl),findall(==(k),fl)
    function emit(β,j,ok)
      ok || error("the path in fold $k did not converge.")
      total=zero(T)
      for i in test;total+=abs2(y[i]-β[i]);end
      errs[k,j]=total/length(test)
    end
    foldpath(train,path.λpath,emit)
  end
  batch=threads ? min(K,max_concurrent_folds) : 1
  for first in 1:batch:K
    nbatch=min(batch,K-first+1)
    _runfolds(k->fold!(first+k-1),nbatch,threads)
  end
  cvm,cvse,jmin,j1se=_cvsummary(errs,T[count(==(k),fl) for k in 1:K])
  CVGeneralizedLassoPath{T}(path,path.λpath,cvm,cvse,path.λpath[jmin],path.λpath[j1se],fl)
end
