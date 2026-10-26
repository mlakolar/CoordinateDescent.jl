using HighDimensionalModels, LinearAlgebra, Statistics, Profile, SHA, TOML
BLAS.set_num_threads(1)
const HDM=HighDimensionalModels
folder,out,variant=ARGS
mkpath(out)
n,p,q=parse.(Int,split(read(joinpath(folder,"dims.txt"),String)))
readarr(name,dims...) = open(io->reshape(read!(io,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name))
X,Y=readarr("X.bin",n,p),readarr("Y.bin",n,q)
label=basename(folder);case,repstr=rsplit(label,'-';limit=2);rep=parse(Int,repstr)
function rss()
    parse(Int,match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String)).captures[1])*1024
end
function coeff_fit(X,Y)
    d=HDM._lr_data(X,Y,true);f=HDM._lr_svd(d)
    HDM._lr_rrcoef(d,f,3)
end
function coeff_path(X,Y)
    d=HDM._lr_data(X,Y,true);f=HDM._lr_svd(d)
    [HDM._lr_rrcoef(d,f,r) for r in 0:3]
end
public_fit(X,Y)=reduced_rank(X,Y,3)
public_path(X,Y)=LowRankPath(X,Y;method=:rank,ranks=0:3)
const MODES=("public_fit","public_path","coefficient_fit","coefficient_path")
functions=(public_fit,public_path,coeff_fit,coeff_path)
function dump_solution(s,r,prefix)
    open(io->write(io,s.B),joinpath(out,"$prefix-$r-B.bin"),"w")
    open(io->write(io,s.residuals),joinpath(out,"$prefix-$r-E.bin"),"w")
    open(io->write(io,s.intercept),joinpath(out,"$prefix-$r-intercept.bin"),"w")
    record=Dict("rank"=>s.rank,"df"=>s.df,"df_exact"=>s.df_exact,"objective"=>s.objective,
                "support"=>s.activerows,"converged"=>s.converged)
    open(io->TOML.print(io,record),joinpath(out,"$prefix-$r.toml"),"w")
end
function persist(value,mode)
    if mode=="public_fit"
        dump_solution(value,3,"fit")
    elseif mode=="public_path"
        for (r,s) in enumerate(value.solutions);dump_solution(s,r-1,"path");end
    elseif mode=="coefficient_fit"
        open(io->write(io,value),joinpath(out,"coefficient-fit.bin"),"w")
    else
        for (r,B) in enumerate(value);open(io->write(io,B),joinpath(out,"coefficient-path-$(r-1).bin"),"w");end
    end
end
for f in functions;f(X,Y);end
meta=Dict("variant"=>variant,"case"=>case,"rep"=>rep,"julia"=>string(VERSION),
    "cpu"=>Sys.cpu_info()[1].model,"blas"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),
    "julia_threads"=>Threads.nthreads(),"warmed_current_rss"=>rss(),"warmed_peak_rss"=>Sys.maxrss(),
    "X_sha256"=>bytes2hex(sha256(read(joinpath(folder,"X.bin")))),
    "Y_sha256"=>bytes2hex(sha256(read(joinpath(folder,"Y.bin")))),
    "source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HDM)),"lowrank.jl")))))
open(joinpath(out,"timings.csv"),"w") do io
    println(io,"variant,case,rep,mode,round,seconds,bytes,gctime")
    for round in 1:3
        for idx in circshift(collect(1:4),mod(rep+round,4))
            GC.gc(); t=@timed functions[idx](X,Y)
            println(io,join((variant,case,rep,MODES[idx],round,t.time,t.bytes,t.gctime),','));flush(io)
            round==1 && persist(t.value,MODES[idx])
        end
    end
end
meta["peak_after_timing"]=Sys.maxrss();meta["current_after_timing"]=rss()
# Component timings are a separate cached-input diagnostic, not an additive decomposition.
d=HDM._lr_data(X,Y,true);f=HDM._lr_svd(d);L,R=HDM._lr_rrfactors(f,3);B=L*R;M=f.U'*d.Y
stages=Dict("centering"=>()->HDM._lr_data(X,Y,true),"design_svd"=>()->svd(d.X;full=false),
 "projection"=>()->f.U'*d.Y,"response_svd"=>()->svd(M;full=false),
 "reconstruction"=>()->(L*R),"dense_residual"=>()->(d.Y-d.X*B),"thin_residual"=>()->(d.Y-(d.X*L)*R),
 "dense_coefficient_rank"=>()->rank(B),"support"=>()->HDM._activerows(B),
 "exact_df"=>()->HDM._lr_exact_df(f,3,q,true))
for fun in values(stages);fun();end
open(joinpath(out,"components.csv"),"w") do io
    println(io,"variant,case,rep,component,round,seconds,bytes")
    for name in sort(collect(keys(stages))),round in 1:3
        t=@timed stages[name]()
        println(io,join((variant,case,rep,name,round,t.time,t.bytes),','))
    end
end
if case in ("responses","wide")
    for idx in (1,2,3)
        Profile.clear();calls=0;began=time()
        Profile.@profile while calls==0 || (calls<20 && time()-began<.25)
            functions[idx](X,Y);calls+=1
        end
        open(joinpath(out,"cpu-$(MODES[idx]).txt"),"w") do io
            println(io,"calls=$calls raw_stack_words=",length(Profile.fetch()))
            Profile.print(io;format=:flat,sortedby=:count)
        end
    end
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=.05 public_fit(X,Y)
    allocations=Profile.Allocs.fetch().allocs
    open(joinpath(out,"alloc-public_fit.txt"),"w") do io
        println(io,"sampled_allocations=",length(allocations))
        for a in sort(collect(allocations);by=x->x.size,rev=true)[1:min(30,length(allocations))]
            println(io,"bytes=",a.size," type=",a.type," stack=",a.stacktrace)
        end
    end
end
meta["whole_worker_peak_rss"]=Sys.maxrss()
open(io->TOML.print(io,meta),joinpath(out,"metadata.toml"),"w")
