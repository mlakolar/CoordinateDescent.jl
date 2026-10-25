include("io.jl")
mkpath(joinpath(ROOT,"inputs"));mkpath(OUT)
meta=open(joinpath(ROOT,"inputs","cases.csv"),"w")
println(meta,"case,name,n,p,density,levels,min_ratio,intercept,standardize_x,baseline")
function savecase(id,name,X,y,L,ratio,intercept,standardize_x;baseline=true)
  n,p=size(X);lambda=CD.slope_bh(p;n)
  Z,v,_,_,_=CD._slope_data(X,y,intercept,standardize_x)
  g=sort(abs.(Z'*v/n);rev=true)
  amax=maximum(cumsum(g)./cumsum(lambda))
  factors=collect(exp.(range(log(amax),log(amax*ratio);length=L)))
  open(joinpath(ROOT,"inputs","case-$id.bin"),"w") do io
    write(io,Int64[n,p,issparse(X),issparse(X) ? nnz(X) : n*p,L,intercept,standardize_x])
    if issparse(X)
      write(io,Int64.(X.colptr));write(io,Int64.(rowvals(X)));write(io,nonzeros(X))
    else
      write(io,X)
    end
    write(io,y);write(io,lambda);write(io,factors)
  end
  println(meta,join((id,name,n,p,issparse(X) ? nnz(X)/(n*p) : 1.0,L,ratio,intercept,standardize_x,baseline),','))
end
# Reuse the exact original four benchmark designs.
open("/home1/mkolar/juliaPkg/runs/20261025-cd82-compare/inputs/bench.bin") do io
  for id in 1:4
    n,p,k,sc,seed,B=Int.(read!(io,zeros(6)))
    X=read!(io,zeros(n,p));y=read!(io,zeros(n))
    read!(io,zeros(n,p));read!(io,zeros(Int32,(n÷2)*2B))
    savecase(id,["dense","wide","correlated","original-sparse"][id],id==4 ? sparse(X) : X,y,20,0.2,false,false)
  end
end
for (id,n,p,d,L,ratio,intercept,stdx,positive,name) in [
  (5,1000,3000,.001,20,.2,false,false,false,"sparse-0.1pct"),
  (6,1000,3000,.01,20,.2,false,false,false,"sparse-1pct"),
  (7,600,1000,.1,20,.2,false,false,false,"sparse-10pct"),
  (8,1000,3000,.01,20,.2,true,true,true,"centered-scaled"),
  (9,1000,3000,.01,50,.02,true,true,true,"long-weak-path"),
  (10,1500,6000,.002,20,.2,true,true,true,"large-centered"),
  (11,10000,20000,.0001,10,.2,true,true,true,"storage-canary")]
  rng=Xoshiro(id==9 ? 9208 : 9200+id)
  X=sprandn(rng,n,p,d)
  positive && (nonzeros(X).=abs.(nonzeros(X)).+0.5)
  # Nontrivial column units, with no zero-column division.
  scales=vec(sqrt.(sum(abs2,X;dims=1)./n));replace!(scales,0.0=>1.0)
  for j in 1:p,k in nzrange(X,j);nonzeros(X)[k]/=scales[j];end
  beta=zeros(p);beta[1:10].=0.4
  y=X*beta+randn(rng,n).+(intercept ? 2.0 : 0.0)
  savecase(id,name,X,y,L,ratio,intercept,stdx;baseline=id!=11)
end
close(meta)
println("GENERATED 11 shared cases")
