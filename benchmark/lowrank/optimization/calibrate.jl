ENV["CD81_LIBRARY"]="1"
include(joinpath(@__DIR__,"julia_worker.jl"))
warm=problem("inputs/small-1")
nuclear("ps_fista",warm,.02)
d=problem("inputs/wide-1")
B,k,ok,detail=nuclear("ps_fista",d,.02)
obj,gap,res=certificate(d,"nuclear",B,.02)
@assert gap<=1e-6 && res<=1e-6
println("CALIBRATION_PASSED ps_fista wide-1 frac=.02 iterations=",k," gap=",gap," residual=",res)
