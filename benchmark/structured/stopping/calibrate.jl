using CoordinateDescent, LinearAlgebra, DelimitedFiles, Statistics, Profile
BLAS.set_num_threads(1)
include("experiment.jl")
include("check-gap.jl")
seed=parse(Int,ARGS[1]);prior=ARGS[2];out=ARGS[3]
settings=Symbol.(split(ARGS[4],','));mkpath(out)
const CD=CoordinateDescent
const MODES=CD._stopping_calibration_mode
table,header=readdlm(joinpath(prior,"inputs/cases.tsv"),'\t',Any;header=true)
function solve(c,y,x)
    levels=c[7]>0 ? exp.(range(log(c[9]),log(c[8]);length=Int(c[7])))./length(y) : [Float64(c[8])/length(y)]
    opts=MODES[]==:strict ? (;abstol=1e-8,reltol=1e-7) : (;abstol=1e-7,reltol=1e-6)
    c[7]>0 ? trend_filter_path(y,levels;order=1,x,opts...) : trend_filter(y,only(levels);order=1,x,opts...)
end

fits(s)=s isa GeneralizedLassoPath ? s.solutions : [s]
function errors(c,s,y,x,reference)
    B=reduce(hcat,getproperty.(fits(s),:x));p,q=size(B)
    D=difference_matrix(p,2;x)
    levels=c[7]>0 ? exp.(range(log(c[9]),log(c[8]);length=q))./p : [Float64(c[8])/p]
    ob=vec(sum(abs2,B.-y;dims=1))./(2p)+levels.*vec(sum(abs,D*B;dims=1))
    refob=vec(sum(abs2,reference.-y;dims=1))./(2p)+levels.*vec(sum(abs,D*reference;dims=1))
    ce=maximum(norm(B[:,j]-reference[:,j])/max(norm(reference[:,j]),sqrt(p)*eps()) for j in 1:q)
    oe=maximum(abs.(ob-refob)./max.(abs.(refob),eps()))
    ce,oe
end
for setting in settings
    MODES[]=setting
    for path in (false,true),x in (nothing,cumsum(fill(1.1,64)))
        path ? trend_filter_path(sin.(1:64),[.1,.01];order=1,x) : trend_filter(sin.(1:64),.01;order=1,x)
    end
end
open(joinpath(out,"calibration-$seed.tsv"),"w") do io
    println(io,"seed\tcase\tsetting\tround\tseconds\tbytes\titerations\tconverged\tcoefficient_error\tobjective_error\tr_coefficient_error\tr_objective_error")
    cases=[collect(c) for c in eachrow(table) if c[2]=="signal" && c[5]==1]
    # Extend the same calibration to the previously measured million-point data.
    push!(cases,Any["million-k1","signal",1000000,1000000,1,"unit",0,1,1,false])
    for c in cases
        n=Int(c[3]);id=string(c[1]);million=id=="million-k1"
        inputs=million ? joinpath(prior,"million/inputs") : joinpath(prior,"inputs")
        refs=million ? joinpath(prior,"million/baseline-full") : joinpath(prior,"baseline-full")
        y=vec(readdlm(joinpath(inputs,string(seed),"signal-$n-y.tsv"),'\t',Float64))
        x=c[6]=="unit" ? nothing : vec(readdlm(joinpath(inputs,string(seed),"irregular-x.tsv"),'\t',Float64))
        q=max(1,Int(c[7]));reference=reshape(read!(joinpath(refs,"tight-$seed-$id.bin"),Vector{Float64}(undef,n*q)),n,q)
        ratios,rh=readdlm(joinpath(prior,"final-report/optimized-selected-ratios.tsv"),'\t',Any;header=true)
        key=million ? "signal-n100000-k1-l1" : id
        row=only(r for r in eachrow(ratios) if r[1]==key)
        rsetting=string(row[findfirst(==("reference"),vec(rh))])
        rfolder=million ? refs : rsetting=="glmgen" ? joinpath(prior,"baseline-full") : joinpath(prior,"reference-loose")
        rbeta=reshape(read!(joinpath(rfolder,"$rsetting-$seed-$id.bin"),Vector{Float64}(undef,n*q)),n,q)
        # Rotate setting order by dataset to reduce fixed-order bias.
        ordered=circshift(settings,seed%length(settings))
        for round in 1:3,setting in ordered
            MODES[]=setting
            GC.gc();m=@timed solve(c,y,x)
            ce,oe=round==1 ? errors(c,m.value,y,x,reference) : (NaN,NaN)
            rce,roe=round==1 ? errors(c,m.value,y,x,rbeta) : (NaN,NaN)
            fs=fits(m.value)
            println(io,join((seed,id,setting,round,m.time,m.bytes,sum(s.iterations for s in fs),
                all(s.converged for s in fs),ce,oe,rce,roe),'\t'));flush(io)
        end
    end
end

# Baseline profiles use ordinary fits, with no reference-error instrumentation.
MODES[]=:baseline
y=vec(readdlm(joinpath(prior,"inputs",string(seed),"signal-100000-y.tsv"),'\t',Float64))
GC.gc();Profile.clear()
Profile.@profile trend_filter(y,1e-5;order=1)
open(joinpath(out,"profile-$seed-linear.txt"),"w") do io
    Profile.print(io;format=:flat,sortedby=:count,mincount=3)
end
