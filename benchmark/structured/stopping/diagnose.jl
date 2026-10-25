# Diagnostic runs only: reference-error checks are excluded from fit timings.
using CoordinateDescent, LinearAlgebra, SparseArrays, DelimitedFiles
BLAS.set_num_threads(1)
const CD = CoordinateDescent
const SEED = parse(Int, ARGS[1])
const PRIOR = ARGS[2]
const OUT = ARGS[3]
mkpath(OUT)

function diagnose(io, id, b, x, alpha, reference; warmup=false)
    n = length(b); A = CD._SignalDesign{Float64}(n)
    K = difference_matrix(n, 1; x)
    if x !== nothing
        K = Diagonal(1 ./ diff(x)) * K
    end
    spacing = x === nothing ? 1.0 : (x[end]-x[1])/(n-1)
    rho = CD._trend_rho(A, 1., n, alpha/n, spacing, 1, Float64)
    F = CD._split_factor(CD._structured_gram(A), K, rho; bandwidth=1)
    g = CD.TotalVariation1D(alpha)
    m = size(K, 1)
    beta=zeros(n); z=zeros(m); u=zeros(m)
    kb=zeros(m); v=zeros(m); zold=zeros(m)
    rhs=zeros(n); dual=zeros(n); ktu=zeros(n)
    q=zeros(m-1); w=zeros(m); dtq=zeros(n)
    refobj=(sum(abs2,reference-b)/2+alpha*sum(abs,diff(K*reference)))/n
    seen=Set{String}()
    function record(kind, label, iteration, pr, dr, gap, obj, ce)
        warmup && return
        println(io,join((SEED,id,kind,label,iteration,pr,dr,gap,obj,
            abs(obj-refobj)/max(abs(refobj),eps()),ce),'\t'));flush(io)
    end
    for iteration in 1:20000
        @. v=z-u
        mul!(rhs,transpose(K),v); @. rhs=b+rho*rhs
        CD._structured_solve!(beta,F,rhs)
        mul!(kb,K,beta);copyto!(zold,z)
        @. v=kb+u
        CD.prox!(z,g,v,inv(rho))
        @. u=u+kb-z
        @. v=kb-z
        pr=norm(v)
        @. v=z-zold
        mul!(dual,transpose(K),v);dr=rho*norm(dual)
        mul!(ktu,transpose(K),u)
        ep=sqrt(m)*1e-7+1e-6*max(norm(kb),norm(z))
        ed=sqrt(n)*1e-7+1e-6*rho*norm(ktu)
        pending=["residual-$(scale)" for scale in (1,2,5) if
            !("residual-$(scale)" in seen) && pr<=scale*ep && dr<=scale*ed]
        strict=pr<=ep/10 && dr<=ed/10
        if iteration%10==0 || !isempty(pending) || strict
            # rho*u belongs to the TV subgradient, up to floating-point error.
            # Clipping its cumulative-sum representation makes q dual-feasible.
            cumulative=0.0
            for j in eachindex(q)
                cumulative+=rho*u[j]
                q[j]=clamp(-cumulative,-alpha,alpha)
            end
            w[1]=-q[1]
            for j in 2:m-1;w[j]=q[j-1]-q[j];end
            w[m]=q[end]
            mul!(dtq,transpose(K),w)
            obj=(sum(abs2,beta-b)/2+alpha*sum(abs,diff(kb)))/n
            # Stable primal-dual gap = stationarity square + TV Fenchel gap.
            gap=(sum(abs2,beta-b+dtq)/2+alpha*sum(abs,diff(kb))-dot(kb,w))/n
            relgap=max(gap,0)/max(abs(obj),eps())
            ce=norm(beta-reference)/max(norm(reference),sqrt(n)*eps())
            record("trace","checkpoint",iteration,pr,dr,relgap,obj,ce)
            for label in pending
                record("candidate",label,iteration,pr,dr,relgap,obj,ce);push!(seen,label)
            end
            for tol in (1e-6,3e-7,1e-7)
                label="gap-$(tol)"
                if !(label in seen) && pr<=5ep && dr<=5ed && relgap<=tol
                    record("candidate",label,iteration,pr,dr,relgap,obj,ce);push!(seen,label)
                end
            end
            if !("target" in seen) && ce<=1e-4 && abs(obj-refobj)<=1e-6*abs(refobj)
                record("candidate","target",iteration,pr,dr,relgap,obj,ce);push!(seen,"target")
            end
            if strict && !("strict" in seen)
                record("candidate","strict",iteration,pr,dr,relgap,obj,ce);push!(seen,"strict")
            end
        end
        length(seen)==8 && break
        warmup && iteration>=100 && break
    end
    warmup || @assert length(seen)==8 (id,seen)
end

diagnose(devnull,"warmup",sin.(1:64),nothing,1.,sin.(1:64);warmup=true)
open(joinpath(OUT,"diagnostic-$SEED.tsv"),"w") do io
    println(io,"seed\tcase\tkind\tsetting\titeration\tprimal\tdual\trelative_gap\tobjective\trelative_objective_error\tcoefficient_error")
    for n in (1000,10000,100000), alpha in (.1,1.,10.)
        id="signal-n$(n)-k1-l$(isinteger(alpha) ? Int(alpha) : alpha)"
        b=vec(readdlm(joinpath(PRIOR,"inputs",string(SEED),"signal-$n-y.tsv"),'\t',Float64))
        reference=read!(joinpath(PRIOR,"baseline-full","tight-$SEED-$id.bin"),Vector{Float64}(undef,n))
        diagnose(io,id,b,nothing,alpha,reference)
    end
    for alpha in (.1,1.,10.)
        n=10000;id="irregular-n$(n)-k1-l$(isinteger(alpha) ? Int(alpha) : alpha)"
        d=joinpath(PRIOR,"inputs",string(SEED))
        b=vec(readdlm(joinpath(d,"signal-$n-y.tsv"),'\t',Float64))
        x=vec(readdlm(joinpath(d,"irregular-x.tsv"),'\t',Float64))
        reference=read!(joinpath(PRIOR,"baseline-full","tight-$SEED-$id.bin"),Vector{Float64}(undef,n))
        diagnose(io,id,b,x,alpha,reference)
    end
    n=1000000;id="million-k1"
    b=vec(readdlm(joinpath(PRIOR,"million/inputs",string(SEED),"signal-$n-y.tsv"),'\t',Float64))
    reference=read!(joinpath(PRIOR,"million/baseline-full","tight-$SEED-$id.bin"),Vector{Float64}(undef,n))
    diagnose(io,id,b,nothing,1.,reference)
end
