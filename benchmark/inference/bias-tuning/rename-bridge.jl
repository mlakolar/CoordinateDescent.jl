# Run from the recorded Discovery run root after restricted development finishes.
using Test, DelimitedFiles
root=pwd()
script=read(joinpath(root,"source-v4","benchmark","inference","bias-tuning","probe-restricted.jl"),String)
script=replace(script,"joinpath(\"restricted\",stage,case)"=>"joinpath(\"rename-bridge\",stage,case)","for rep=shard:8:maxrep"=>"for rep=1:maxrep")
for family in ["gaussian","binomial"],kind in ["moderate","correlated","benign","random"]
 empty!(ARGS);append!(ARGS,["development",family*"-"*kind,"1","3"])
 include_string(Main,script,"rename-bridge-probe.jl")
end
function records(path)
 lines=readlines(path)
 Dict((s[2],parse(Int,s[3]))=>s for line in lines[2:end] for s in [split(line,',')])
end
largest=0.;fits=0
@testset "Renamed package and current master dependency bridge" begin
 for family in ["gaussian","binomial"],kind in ["moderate","correlated","benign","random"],rep in 1:3
  case=family*"-"*kind
  old=records(joinpath(root,"restricted","development",case,"$rep.coefficients"))
  new=records(joinpath(root,"rename-bridge","development",case,"$rep.coefficients"))
  @test keys(old)==keys(new)
  for key in keys(old),k in 4:11
   a=parse(Float64,old[key][k]);b=parse(Float64,new[key][k])
   @test (isnan(a)&&isnan(b)) || isapprox(a,b;atol=1e-7,rtol=1e-7)
   if isfinite(a)&&isfinite(b);global largest=max(largest,abs(a-b));end
  end
  global fits+=length(Set(k[1] for k in keys(new)))
 end
end
println("RENAME_BRIDGE fits=",fits," datasets=24 maximum_absolute_difference=",largest)
