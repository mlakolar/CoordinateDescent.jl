# Exact regeneration of saved own-arm campaign data; checks original SHA256 before export.
using SHA
include("confirmation-data.jl")
family==:gaussian || error("Gaussian only")
for rep=shard:8:maxrep
 original=joinpath("outputs","campaign",caseid,string(rep)*".gaussian.diagnostics.csv")
 header=split(readline(original),',');row=split(readlines(original)[2],',')
 dest=joinpath("competitor-data",caseid,string(rep))
 if rep<=20 && isfile(joinpath(dest,"hashes.txt")) && all(isfile(joinpath(dest,nm*".csv")) for nm in ["X","y","truth","targets"])
  expected="X="*row[findfirst(==("input_x_sha256"),header)]*"\ny="*row[findfirst(==("input_y_sha256"),header)]*"\n"
  read(joinpath(dest,"hashes.txt"),String)==expected || error("Existing pilot hash receipt mismatch")
  continue
 end
 X,y,b,targets=dataset(rep)
 xh=bytes2hex(sha256(reinterpret(UInt8,vec(X))));yh=bytes2hex(sha256(reinterpret(UInt8,y)))
 xh==row[findfirst(==("input_x_sha256"),header)] && yh==row[findfirst(==("input_y_sha256"),header)] || error("Campaign input hash mismatch")
 dest=joinpath("competitor-data",caseid,string(rep));mkpath(dest)
 for (name,v) in [("X",X),("y",y),("truth",b),("targets",targets)];writedlm(joinpath(dest,name*".csv"),v,',');end
 write(joinpath(dest,"hashes.txt"),"X="*xh*"\ny="*yh*"\n")
end
