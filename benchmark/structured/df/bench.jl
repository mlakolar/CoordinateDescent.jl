using HighDimensionalModels, LinearAlgebra, SparseArrays, Random, Statistics, Profile, Serialization, SHA
BLAS.set_num_threads(1)
variant, repstr, out = ARGS
rep = parse(Int,repstr)
mkpath(out)
HDM = HighDimensionalModels
timings = open(joinpath(out,"timings-$variant-$rep.csv"),"w")
println(timings,"variant,rep,case,kind,round,seconds,bytes,gctime,converged,iterations,df,pattern_count")
metadata = open(joinpath(out,"metadata-$variant-$rep.txt"),"w")
println(metadata,"Julia=",VERSION,"\nCPU=",Sys.cpu_info()[1].model,"\nBLAS=",BLAS.get_config(),"\nthreads=",Threads.nthreads(),"/",BLAS.get_num_threads())
println(metadata,"source_sha256=",bytes2hex(sha256(read(joinpath(dirname(pathof(HDM)),"structured.jl")))))
function rss()
    matchresult = match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String))
    parse(Int,matchresult.captures[1])*1024
end
function dense_df(A,D,b,tol)
    inactive = findall(v->abs(v)<=tol,D*b)
    N = isempty(inactive) ? Matrix{eltype(b)}(I,length(b),length(b)) : nullspace(Matrix(D[inactive,:]))
    size(N,2)==0 ? 0 : rank(A*N)
end
cases = [("fused40",200,40,:fused,50),("fused100",600,100,:fused,50),
         ("fused300",1200,300,:fused,30),("graph100",400,100,:graph,20),
         ("trend40",200,40,:trend,50),("trend100",500,100,:trend,20),
         ("rankdef100",240,100,:rankdef,20),("wide100",60,100,:fused,20)]
for (ci,(label,n,p,family,nlevels)) in enumerate(cases)
    rng = Xoshiro(990000+100rep+ci)
    X = randn(rng,n,p)
    for j in 2:p
        @views X[:,j] .= .4X[:,j-1] .+ sqrt(1-.4^2).*X[:,j]
    end
    family==:rankdef && (X[:,51:end] .= X[:,1:50])
    b = repeat([0.,1.,-1.,0.];inner=cld(p,4))[1:p]
    y = X*b + .5randn(rng,n)
    levels = exp.(range(log(.5),log(.003);length=nlevels))
    D = family==:graph ? graph_difference(p,[(j,j+1) for j in 1:p-1];weights=0.5 .+ rand(rng,p-1)) :
        family==:trend ? difference_matrix(p,2) : [difference_matrix(p);spdiagm(0=>ones(p))]
    fitpath() = family==:graph ? GeneralizedLassoPath(X,y,D,levels;rho=100.,max_iter=40_000) :
        family==:trend ? trend_filter_path(X,y,levels;order=1,max_iter=40_000) :
        fused_lasso_path(X,y,levels;max_iter=40_000)
    # Compilation, loading and independent correctness checking are outside timers.
    warm = fitpath()
    println(metadata,"$label warmed_rss_bytes=",rss())
    flush(metadata)
    for round in 1:3
        GC.gc()
        measured = @timed fitpath()
        path = measured.value
        println(timings,join((variant,rep,label,"path",round,measured.time,measured.bytes,measured.gctime,
            all(s->s.converged,path.solutions),sum(s.iterations for s in path.solutions),sum(s.df for s in path.solutions),0),','))
        flush(timings)
    end
    @assert all(s->s.converged,warm.solutions) "$label convergence"
    expected = [dense_df(X,D,s.x,1e-4) for s in warm.solutions]
    @assert expected == [s.df for s in warm.solutions] "$label dense DF disagreement"
    state = (x=coef(warm),objective=[s.objective for s in warm.solutions],df=expected,
        iterations=[s.iterations for s in warm.solutions],converged=[s.converged for s in warm.solutions])
    serialize(joinpath(out,"fit-$variant-$rep-$label.bin"),state)
    # Repeat the actual path patterns, measuring diagnostic work and cache setup separately.
    patterns = length(unique([findall(v->abs(v)<=1e-4,D*s.x) for s in warm.solutions]))
    setup() = variant=="candidate" ? HDM._StructuredDFWorkspace(X,D) : nothing
    setup()
    for round in 1:3
        measured = @timed setup()
        ws = measured.value
        println(timings,join((variant,rep,label,"df_setup",round,measured.time,measured.bytes,measured.gctime,true,0,0,patterns),','))
        # Fresh cache per round; include every path pattern exactly once.
        evaldf() = variant=="candidate" ? [HDM._structured_df(X,D,s.x,1e-4;workspace=ws) for s in warm.solutions] :
            [HDM._structured_df(X,D,s.x,1e-4) for s in warm.solutions]
        evaldf()  # compile the diagnostic closure outside timing
        ws !== nothing && (ws.valid=false)
        measured = @timed evaldf()
        @assert measured.value==expected
        println(timings,join((variant,rep,label,"df",round,measured.time,measured.bytes,measured.gctime,true,0,sum(expected),patterns),','))
    end
    # CPU and allocation profiles are separate from fit timing.
    Profile.clear()
    Profile.@profile begin
        deadline=time()+.25
        calls=0
        while calls<100 && (calls==0 || time()<deadline)
            fitpath()
            calls+=1
        end
    end
    open(joinpath(out,"cpu-$variant-$rep-$label.txt"),"w") do io
        println(io,"raw_stack_words=",length(Profile.fetch()))
        Profile.print(io;format=:flat,sortedby=:count)
    end
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=.05 fitpath()
    allocations=Profile.Allocs.fetch().allocs
    open(joinpath(out,"alloc-$variant-$rep-$label.txt"),"w") do io
        println(io,"sampled_allocations=",length(allocations))
        for a in sort(collect(allocations);by=x->x.size,rev=true)[1:min(30,length(allocations))]
            println(io,"bytes=",a.size," type=",a.type," stack=",a.stacktrace)
        end
    end
    println(metadata,"$label peak_after_profile_bytes=",Sys.maxrss())
    flush(timings)
end
# Large-p DF-only diagnostic; public path scaling is reported separately.
let rng=Xoshiro(999000+rep), p=1000, n=1200
    X=randn(rng,n,p); D=difference_matrix(p)
    b=repeat(randn(rng,100);inner=10)
    expected=dense_df(X,D,b,1e-4)
    for round in 1:3
        measured=@timed begin
            ws=variant=="candidate" ? HDM._StructuredDFWorkspace(X,D) : nothing
            variant=="candidate" ? HDM._structured_df(X,D,b,1e-4;workspace=ws) : HDM._structured_df(X,D,b,1e-4)
        end
        @assert measured.value==expected
        println(timings,join((variant,rep,"graph1000","df_including_setup",round,measured.time,measured.bytes,measured.gctime,true,0,expected,1),','))
    end
end
println(metadata,"worker_peak_rss_bytes=",Sys.maxrss())
close(metadata); close(timings)
