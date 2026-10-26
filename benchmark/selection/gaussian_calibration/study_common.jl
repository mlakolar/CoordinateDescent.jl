using LinearAlgebra, Random, Statistics, SparseArrays, SHA, TOML, DelimitedFiles
import HighDimensionalModels as H
import SLOPE as SJ
BLAS.set_num_threads(1)
const STUDY = ENV["HDM_STUDY"]
const METHODS = ["bh", "paper", "julia"]
const SCENARIOS = ["iid-alt","iid-same","ar04-alt","ar04-same","ar08-alt","ar08-same",
 "weak-ar04-alt","weak-ar04-same","wide-ar04-alt","wide-ar04-same","null-ar04",
 "orth-alt","orth-same","estimated-iid","estimated-ar04","block-alt"]
function config(sc)
    n,p = sc in (9,10) ? (180,300) : (300,100)
    rho = sc in (1,2,12,13,14) ? 0.0 : sc in (5,6) ? 0.8 : 0.4
    (;n,p,rho,k=sc==11 ? 0 : 20,amp=sc in (7,8) ? 3.0 : 4.5,
      same=sc in (2,4,6,8,10,13),intercept=sc in (12,13),estimated=sc in (14,15),block=sc==16)
end
function readcase(io)
    eof(io) && return nothing
    n,p,k,sc,seed,intercept,dataset,nweights = Int.(read!(io,Vector{Int64}(undef,8)))
    sigma=read(io,Float64)
    X=read!(io,Matrix{Float64}(undef,n,p)); y=read!(io,Vector{Float64}(undef,n))
    beta=read!(io,Vector{Float64}(undef,p)); W=read!(io,Matrix{Float64}(undef,p,nweights))
    (;n,p,k,sc,seed,intercept=Bool(intercept),dataset,sigma,X,y,beta,W)
end
function centered(c)
    xb=c.intercept ? vec(mean(c.X;dims=1)) : zeros(c.p)
    yb=c.intercept ? mean(c.y) : 0.0
    c.X .- xb', c.y .- yb, xb, yb
end
function writerecord(io,c,W,B)
    write(io,Int64[c.seed,c.p,size(B,2)]);write(io,Matrix(W));write(io,Matrix(B));flush(io)
end
function provenance(out,engine,input)
    d=Dict("engine"=>engine,"julia"=>string(VERSION),"cpu"=>Sys.cpu_info()[1].model,
      "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
      "input_sha256"=>bytes2hex(sha256(read(input))),"SLOPE"=>string(pkgversion(SJ)),
      "hdm_slope_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"slope.jl")))),
      "sj_regweights_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(SJ)),"regweights.jl")))))
    open(io->TOML.print(io,d),out,"w")
end
