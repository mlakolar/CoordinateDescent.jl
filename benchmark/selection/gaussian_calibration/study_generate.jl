include("study_common.jl")
mode=ARGS[1]; shard=parse(Int,ARGS[2]); B=parse(Int,ARGS[3])
mode in ("pilot","main") || error("mode")
mode=="main" && !(1<=shard<=7 && B==144) && error("declared cohort mismatch")
mkpath(joinpath(STUDY,"inputs"))
for sc in eachindex(SCENARIOS)
    cfg=config(sc);n,p=cfg.n,cfg.p
    filename=joinpath(STUDY,"inputs","sc$sc-shard$shard.bin")
    ispath(filename) && error("immutable input already exists: $filename")
    open(filename,"w") do io
        for dataset in 1:B
            seed=mode=="main" ? 107400000+1000000sc+10000shard+dataset : 107000000+10000sc+100shard+dataset
            rng=Xoshiro(seed)
            if cfg.intercept
                Z=randn(rng,n,p);Z .-= mean(Z;dims=1)
                X=sqrt(n)*Matrix(qr(Z).Q)[:,1:p]
            else
                S=[cfg.rho^abs(i-j) for i in 1:p,j in 1:p]
                if cfg.block
                    h=p÷2
                    S=[(i<=h)==(j<=h) ? (i<=h ? 0.8 : -0.4)^abs(i-j) : 0.0 for i in 1:p,j in 1:p]
                end
                X=randn(rng,n,p)*cholesky(Symmetric(S)).U
            end
            beta=zeros(p)
            beta[1:cfg.k].=cfg.amp/sqrt(n).*(cfg.same ? ones(cfg.k) : ifelse.(isodd.(1:cfg.k),1.0,-1.0))
            y=X*beta+randn(rng,n);sigma=cfg.estimated ? std(y) : 1.0
            W=hcat(H.slope_bh(p;n,q=0.1,sigma),H.slope_gaussian(p;n,q=0.1,sigma,intercept=cfg.intercept),
                sigma/sqrt(n)*SJ.regweights(p;n=n-Int(cfg.intercept),q=0.1,type=:gaussian))
            write(io,Int64[n,p,cfg.k,sc,seed,cfg.intercept,dataset,3]);write(io,sigma)
            write(io,X);write(io,y);write(io,beta);write(io,W)
        end
    end
    open(filename*".sha256","w") do io
        println(io,bytes2hex(sha256(read(filename))),"  ",basename(filename))
    end
    println("GENERATED ",SCENARIOS[sc]," ",shard," ",B);flush(stdout)
end
