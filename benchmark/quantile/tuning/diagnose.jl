using HighDimensionalModels, HiGHS, Random, Statistics, LinearAlgebra, DelimitedFiles
const HDM=HighDimensionalModels
BLAS.set_num_threads(1)
loadbin(name)=open(io->read!(io,Vector{Float64}(undef,filesize("reference/$name.bin")÷8)),"reference/$name.bin")
X=reshape(loadbin("X"),83,5);y=loadbin("y");o=loadbin("offset");w=[.7,1.,1.2,.8,2.]
fl=[i%3+1 for i in 1:83];ls=[.3,.15,.05]
refs=readdlm("reference/calibration.csv",',',Any;skipstart=1)
mkpath("results")
open("results/tied-solutions.csv","w") do io
 println(io,"tau,standardized,fold,level,training_n,max_coefficient_difference,objective_difference,R_kkt,R_gap,API_kkt,API_gap,heldout_loss_difference")
 for row in eachrow(refs)
  tau=Float64(row[1]);scaled=lowercase(string(row[2]))=="true";key="$(tau)-$(scaled ? "TRUE" : "FALSE")"
  for k in 1:3
   tr,te=fl.!=k,fl.==k;A=X[tr,:];z=y[tr].-o[tr];nt=sum(tr)
   B=reshape(loadbin("$key-fold$k-coef"),6,3);U=reshape(loadbin("$key-fold$k-dual"),nt,3)
   fp=QuantileLassoPath(A,z,ls;tau,standardize_x=scaled,penalty_factors=w,tol=2e-6)
   effective=scaled ? w.*vec(sqrt.(mean((A.-mean(A;dims=1)).^2;dims=1))) : w
   for j in 1:3
    ref=HDM._qsolution(A,z,B[2:end,j],B[1,j],U[:,j],ls[j],effective,tau,true,:exact,0.,:R_returned,0,true,2e-6)
    train_difference=abs(ref.objective-fp[j].objective)
    rR=y[te].-o[te].-B[1,j].-X[te,:]*B[2:end,j]
    rA=y[te].-predict(fp[j],X[te,:];offset=o[te])
    lossdiff=abs(mean(max.(tau.*rR,(tau-1).*rR))-mean(max.(tau.*rA,(tau-1).*rA)))
    println(io,join((tau,scaled,k,j,nt,maximum(abs.(B[:,j].-coef(fp[j]))),train_difference,ref.kkt_residual,ref.duality_gap,fp[j].kkt_residual,fp[j].duality_gap,lossdiff),','))
   end
  end
 end
end
println("TIED SOLUTION DIAGNOSTICS COMPLETE")
