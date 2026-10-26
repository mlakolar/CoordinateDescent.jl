using LinearAlgebra
root=ENV["HDM_KNOCKOFF_RUNTIME"]
for input in 1:3
 out=joinpath(root,"candidate-case-5-rep-0")
 open(joinpath(out,"input-$input.bin")) do io
  n,p,seed=read!(io,Vector{Int64}(undef,3));X=read!(io,Matrix{Float64}(undef,n,p))
  F=X./sqrt.(sum(abs2,X;dims=1));path=joinpath(out,"unitnorm-$input.bin")
  ispath(path) && error("immutable unitnorm input exists")
  open(dst->write(dst,F),path,"w")
 end
end
