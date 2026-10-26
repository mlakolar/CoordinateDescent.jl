using HighDimensionalModels,ProximalAlgorithms,LinearAlgebra,DelimitedFiles,TOML,SHA
BLAS.set_num_threads(1)
root="results/prototype-canary-v2";mkpath(root);records=Any[]
for case in ("small","medium","responses","wide","ill","deficient")
 folder=joinpath("inputs/pilot-v1",case);a=vec(readdlm(joinpath(folder,"dims.txt")));n,p,q=Int.(a[1:3]);lm=a[4]
 readarr(name,dims...)=open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name*".bin"))
 X,Y=readarr("X",n,p),readarr("Y",n,q)
 out=joinpath(root,case);mkpath(out)
 for fraction in (.5,.1,.02)
  s=nuclear_regression(X,Y,fraction*lm;solver=:panoc,tol=1e-8,max_iter=20000)
  s.converged || error("candidate cold not converged $case/$fraction")
  file=joinpath(out,"cold-$fraction.bin");open(f->write(f,s.B),file,"w")
  push!(records,Dict("case"=>case,"kind"=>"cold","fraction"=>fraction,"lambda"=>fraction*lm,"file"=>file,"converged"=>s.converged,"iterations"=>s.iterations,"objective"=>s.objective,"rank"=>s.rank,"df"=>s.df))
 end
 fractions=exp.(range(log(.5),log(.02);length=8))
 path=LowRankPath(X,Y;solver=:panoc,λpath=fractions.*lm,tol=1e-8,max_iter=20000)
 for (j,s) in enumerate(path.solutions)
  s.converged || error("candidate path not converged $case/$j")
  file=joinpath(out,"path-$j.bin");open(f->write(f,s.B),file,"w")
  push!(records,Dict("case"=>case,"kind"=>"path","fraction"=>fractions[j],"lambda"=>fractions[j]*lm,"file"=>file,"converged"=>s.converged,"iterations"=>s.iterations,"objective"=>s.objective,"rank"=>s.rank,"df"=>s.df))
 end
end
open(f->TOML.print(f,Dict("records"=>records,"extension_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HighDimensionalModels)),"..","ext","HighDimensionalModelsProximalAlgorithmsExt.jl")))))),joinpath(root,"fits.toml"),"w")
