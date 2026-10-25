include("io.jl")
rep=parse(Int,ARGS[1])
ids=parse.(Int,split(get(ENV,"CASES","1,2,3,4,5,6,7,8,9,10,11"),','))
open(joinpath(OUT,"accuracy-$rep.csv"),"w") do io
 println(io,"rep,case,engine,relative_residual,relative_objective_difference,max_prediction_difference,max_coefficient_difference")
 for id in ids
  c=loadcase(id);native=readdlm(joinpath(OUT,"coef-$rep-$id-native.csv"),',',Float64)
  _,ref=certificate(c,native[2:end,:],native[1,:])
  for engine in ("native","baseline","SLOPE.jl","R")
   engine=="baseline" && id==11 && continue
   B=readdlm(joinpath(OUT,"coef-$rep-$id-$engine.csv"),',',Float64)
   r,obj=certificate(c,B[2:end,:],B[1,:])
   objerr=maximum(abs.(obj-ref)./max.(1.0,abs.(ref)))
   pdiff=maximum(abs,c.X*(B[2:end,:]-native[2:end,:]).+transpose(B[1,:]-native[1,:]))
   println(io,join((rep,id,engine,r,objerr,pdiff,maximum(abs,B-native)),','));flush(io)
   r<=1e-7 || error("$engine $id residual $r")
   objerr<=1e-7 || error("$engine $id objective $objerr")
  end
 end
end
println("AUDIT_DONE rep=$rep")
