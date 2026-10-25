# Common objective and coefficient checks, outside measured fitting time.
using DelimitedFiles, LinearAlgebra, Statistics, SparseArrays, CoordinateDescent
BLAS.set_num_threads(1)
input, output, folders... = ARGS
tab,h = readdlm(joinpath(input,"cases.tsv"),'\t',Any;header=true)
mkpath(output)
function readbeta(path,p,q)
  v=read!(path,Vector{Float64}(undef,p*q)); reshape(v,p,q)
end
function objective(B,y,X,D,levels)
  R = (X===nothing ? B : X*B) .- y
  vec(sum(abs2,R;dims=1))./(2length(y)) .+ levels .* vec(sum(abs,D*B;dims=1))
end
open(joinpath(output,"accuracy.tsv"),"w") do io
  println(io,"seed\tcase\tcomparison\tmax_relative_coefficient_error\tmax_relative_objective_difference\tfinite\tcoefficient_pass\tobjective_pass")
  for seed in 8201:8203, row in eachrow(tab)
    id,family,n,p,k,grid,path,lo,hi,_ = row[1:10]
    n,p,k,path=Int(n),Int(p),Int(k),Int(path)
    q=max(path,1); levels = (path>0 ? exp.(range(log(hi),log(lo);length=path)) : [lo])./n
    dir=joinpath(input,string(seed))
    if family=="signal"
      y=vec(readdlm(joinpath(dir,"signal-$n-y.tsv"),'\t',Float64)); X=nothing
      x=grid=="unit" ? nothing : vec(readdlm(joinpath(dir,"irregular-x.tsv"),'\t',Float64))
      D=difference_matrix(p,k+1;x)
    else
      X=readdlm(joinpath(dir,"reg-$p-X.tsv"),'\t',Float64); y=vec(readdlm(joinpath(dir,"reg-$p-y.tsv"),'\t',Float64))
      D=family=="fused" ? [.25spdiagm(0=>ones(p)); .75difference_matrix(p)] : family=="trend" ? difference_matrix(p,2) :
        graph_difference(p,[(j,j+1) for j in 1:p-1] ∪ [(j,p-j+1) for j in 1:p÷5])
    end
    found=Dict{String,Matrix{Float64}}()
    for folder in folders
      for solver in ("julia","glmgen","glmgen-default","glmgen-1e8","genlasso","genlasso-qr","genlasso-special","tight")
        file=joinpath(folder,"$solver-$seed-$id.bin")
        isfile(file) && (found[solver]=readbeta(file,p,q))
      end
    end
    for (a,b) in (("julia","glmgen"),("julia","genlasso"),("glmgen","genlasso"),("julia","tight"),("glmgen","tight"),("julia","genlasso-special"),("genlasso-special","genlasso"),
                  ("julia","glmgen-default"),("julia","glmgen-1e8"),("glmgen-default","tight"),("glmgen-1e8","tight"),
                  ("julia","genlasso-qr"),("genlasso-qr","genlasso-special"))
      haskey(found,a) && haskey(found,b) || continue
      A,B=found[a],found[b]
      err=maximum(norm(A[:,j]-B[:,j])/max(norm(B[:,j]),sqrt(p)*eps()) for j in 1:q)
      oa,ob=objective(A,y,X,D,levels),objective(B,y,X,D,levels)
      oe=maximum(abs.(oa.-ob)./max.(abs.(ob),eps()))
      println(io,join((seed,id,"$a/$b",err,oe,all(isfinite,A)&&all(isfinite,B),err<=1e-4,oe<=1e-6),'\t'))
    end
  end
end
