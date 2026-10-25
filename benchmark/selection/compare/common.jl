using LinearAlgebra, Random, Statistics, SparseArrays, DelimitedFiles
import CoordinateDescent as CD
import SLOPE as SJ
import Knockoffs as KJ
import GLMNet as GN
import ProximalBase as PB
BLAS.set_num_threads(1)
const ROOT = ENV["RUN"]
const OUT = joinpath(ROOT,"results")
const BH = CD.slope_bh
const SCENARIOS = ["iid", "ar04", "ar08", "weak-ar04", "null-ar04", "wide-ar04"]
const MC_B = parse(Int,get(ENV,"MC_B","20"))
const NPAIRS = 30

function writecase(io,X,y,Z,pairs,k,sc,seed)
    n,p=size(X); B=length(pairs)
    write(io,Float64[n,p,k,sc,seed,B]); write(io,Matrix(X));write(io,y);write(io,Z)
    rows=hcat((a for pair in pairs for a in pair)...)
    write(io,Int32.(rows))
end
function readcase(io)
    eof(io) && return nothing
    n,p,k,sc,seed,B=Int.(read!(io,Vector{Float64}(undef,6)))
    X=read!(io,Matrix{Float64}(undef,n,p));y=read!(io,Vector{Float64}(undef,n))
    Z=read!(io,Matrix{Float64}(undef,n,p));rows=read!(io,Matrix{Int32}(undef,n÷2,2B))
    pairs=[(Int.(rows[:,2b-1]),Int.(rows[:,2b])) for b in 1:B]
    (;X,y,Z,pairs,k,sc,seed,n,p)
end
sigma(p,rho) = [rho^abs(i-j) for i in 1:p,j in 1:p]
function generate(rep)
    mkpath(joinpath(ROOT,"inputs"));mkpath(OUT)
    file=joinpath(ROOT,"inputs","mc-$rep.bin")
    open(file,"w") do io
        for sc in eachindex(SCENARIOS), b in 1:MC_B
            seed=8300000+100000rep+1000sc+b
            rng=Xoshiro(seed); n,p=sc==6 ? (180,300) : (300,100)
            rho=sc==1 ? 0.0 : sc==3 ? 0.8 : 0.4
            X=randn(rng,n,p)*cholesky(sigma(p,rho)).U
            k=sc==5 ? 0 : 20; amp=(sc==4 ? 3.0 : 4.5)/sqrt(n)
            beta=zeros(p);beta[1:k].=amp.*ifelse.(isodd.(1:k),1,-1)
            y=X*beta+randn(rng,n)
            pairs=CD.complementary_pairs(n,NPAIRS;rng)
            writecase(io,X,y,randn(rng,n,p),pairs,k,sc,seed)
        end
    end
    if rep==1
        open(joinpath(ROOT,"inputs","bench.bin"),"w") do io
            for (sc,(n,p,rho,sp)) in enumerate([(600,200,0.3,false),(200,1000,0.3,false),(400,400,0.8,false),(600,1000,0.0,true)])
                rng=Xoshiro(8300+sc)
                X=sp ? Matrix(sprandn(rng,n,p,0.02)) : randn(rng,n,p)*cholesky(sigma(p,rho)).U
                X ./= sqrt.(sum(abs2,X;dims=1)./n)
                y=X[:,1:10]*fill(0.4,10)+randn(rng,n)
                pairs=CD.complementary_pairs(n,NPAIRS;rng)
                writecase(io,X,y,randn(rng,n,p),pairs,10,sc,8300+sc)
            end
        end
    end
end

function slope_cd(X,y,λ,factors)
    f=CD.SLOPEPath(X,y,factors;lambda=λ,tol=1e-7,max_iter=100000)
    all(f.converged) || error("CD SLOPE convergence")
    hcat(f.βpath...)
end
function slope_sj(X,y,λ,factors)
    f=SJ.slope(X,y;α=factors,λ,fit_intercept=false,centering=:none,scaling=:none,
        tol=1e-8,max_it=100000,max_clusters=size(X,2)+1,dev_change_tol=0.0,dev_ratio_tol=1.0)
    length(f.coefficients)==length(factors) || error("SLOPE.jl truncated path")
    hcat((vec(Matrix(b)) for b in f.coefficients)...)
end
function path_factors(X,y,λ)
    g=sort(abs.(X'*y/length(y));rev=true)
    amax=maximum(cumsum(g)./cumsum(λ))
    collect(exp.(range(log(amax),log(0.2amax);length=20)))
end
function slope_accuracy(X,y,betas,λ,factors)
    length(factors)==size(betas,2) || error("path dimensions")
    maxres=0.0;objs=Float64[]
    scale=max(1.0,norm(X'*y/length(y),Inf))
    for j in eachindex(factors)
        b=betas[:,j];weights=factors[j]*λ;g=X'*(X*b-y)/length(y)
        z=similar(b);PB.prox!(z,PB.SLOPE(weights),b-g,1.0)
        maxres=max(maxres,norm(b-z,Inf)/scale)
        push!(objs,sum(abs2,X*b-y)/(2length(y))+dot(weights,sort(abs.(b);rev=true)))
    end
    maxres,objs
end
function lasso_gn(X,y,lambda)
    fit=GN.glmnet(X,y;lambda=[lambda],intercept=false,standardize=false,tol=1e-14,maxit=1000000)
    vec(convert(Matrix{Float64},fit.betas))
end
function lasso_cd(X,y,lambda)
    f=CD.lasso(X,y,lambda;intercept=false,standardize_x=false,opt_tol=1e-9,max_iter=100000)
    f.converged || error("CD lasso convergence")
    Vector(f.x)
end
function wdifference(b)
    p=length(b)÷2
    abs.(b[1:p])-abs.(b[p+1:end])
end
function selected_w(W;q=0.1)
    t=CD.knockoff_threshold(W;q)
    findall(>=(t),W)
end
function cpss_gn(X,y,lambda,pairs,cap)
    p=size(X,2);sel=falses(p,2length(pairs))
    for (j,rows) in enumerate(a for pair in pairs for a in pair)
        b=abs.(lasso_gn(X[rows,:],y[rows],lambda));order=sortperm(b;rev=true)
        sel[order[1:min(cap,count(>(0),b))],j].=true
    end
    vec(mean(sel;dims=2))
end
function cpss_cd(X,y,lambda,pairs,cap;parallel=false)
    CD.stability_selection(X,y,lambda;pairs,q=cap,cutoff=0.8,parallel,
        intercept=false,standardize_x=false,opt_tol=1e-9,max_iter=100000,rng=Xoshiro(82)).frequencies
end
function record(io,c,method,selected;lambda=NaN,cap=0,bound=NaN)
    V=count(>(c.k),selected);R=length(selected)
    pow=c.k==0 ? NaN : (R-V)/c.k
    println(io,join((c.sc,c.seed,c.n,c.p,c.k,method,lambda,cap,bound,R,V,V/max(1,R),pow,join(selected,';')),','));flush(io)
end
