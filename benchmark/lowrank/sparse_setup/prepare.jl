using HighDimensionalModels, LinearAlgebra, TOML, SHA
BLAS.set_num_threads(1)
function prepare(root, reps)
 for rep in reps, name in ("small","medium","responses","wide","ill","deficient")
  folder=joinpath(root,"$name-$rep");n,p,q=parse.(Int,split(read(joinpath(folder,"dims.txt"),String)))
  readarr(key,dims...)=open(io->reshape(read!(io,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,"$key.bin"))
  X,Y=readarr("X",n,p),readarr("Y",n,q)
  d=HighDimensionalModels._lr_data(X,Y,true);f=HighDimensionalModels._lr_svd(d)
  B=HighDimensionalModels._lr_rrcoef(d,f,3);S=svd(B;full=false);V=S.V[:,1:3];A=B*V
  adaptive=HighDimensionalModels._lr_weights(d,B,nothing,true,2)
  C=d.X'd.Y/n
  for (key,z) in (("B0",B),("A0",A),("V0",V),("adaptive",adaptive))
    open(io->write(io,z),joinpath(folder,"$key.bin"),"w")
  end
  m=Dict("lambda_unit"=>maximum(norm(view(C,j,:)) for j in 1:p),
    "lambda_adaptive"=>maximum(norm(view(C,j,:))/adaptive[j] for j in 1:p),
    "L"=>(1+sqrt(eps(Float64)))*first(f.s)^2/n,
    "baseline_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HighDimensionalModels)),"lowrank.jl")))))
  open(io->TOML.print(io,m),joinpath(folder,"prepared.toml"),"w")
  open(joinpath(folder,"scalars.txt"),"w") do io
    println(io,m["lambda_unit"]);println(io,m["lambda_adaptive"]);println(io,m["L"])
  end
 end
end
prepare(ARGS[1], length(ARGS)>1 && ARGS[2]=="main" ? (1:20) : (0:0))
