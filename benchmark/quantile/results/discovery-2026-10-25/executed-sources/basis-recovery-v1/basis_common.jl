include("solvers.jl")
include("null_basis.jl")
include("split_common.jl")
function readbinary(file)
 open(file) do io;read!(io,Vector{Float64}(undef,filesize(file)÷8));end
end
function runmethod(name,X,y,ls;limit=30.)
 w=ones(size(X,2))
 name=="lp-dual-nullbasis-warm" && return nullpath(X,y,ls,.5,w;limit)
 name=="lp-dual-dual-warm" && return lppath(X,y,ls,.5,w,:dual,:dual,true;limit)
 splitpath(X,y,ls,.5,w,name=="lp-split-primal-warm" ? :primal : :dual;limit)
end
