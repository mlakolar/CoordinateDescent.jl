include("study_common.jl")
mode=ARGS[1];rep=parse(Int,ARGS[2]);mode in ("pilot","main") || error("mode")
mkpath(joinpath(STUDY,"inputs"))
for (sc,(n,p,rho,sp)) in enumerate([(600,200,.3,false),(200,1000,.3,false),(400,400,.8,false),(600,1000,0.,true)])
    filename=joinpath(STUDY,"inputs","sc$sc-shard$rep.bin")
    ispath(filename) && error("immutable runtime inputs exist")
    open(filename,"w") do io
      for dataset in 1:3
        seed=mode=="main" ? 109400000+1000000sc+10000rep+dataset : 109000000+10000sc+100rep+dataset
        rng=Xoshiro(seed)
        X=sp ? Matrix(sprandn(rng,n,p,.02)) : randn(rng,n,p)*cholesky(Symmetric([rho^abs(i-j) for i in 1:p,j in 1:p])).U
        scales=sqrt.(sum(abs2,X;dims=1)./n)
        all(>(0),scales) || error("zero runtime column; retain this input-generation failure")
        X ./= scales
        beta=zeros(p);beta[1:10].=.4;y=X*beta+randn(rng,n)
        W=hcat(H.slope_bh(p;n),H.slope_gaussian(p;n),SJ.regweights(p;n,q=.1,type=:gaussian)/sqrt(n))
        write(io,Int64[n,p,10,sc,seed,0,dataset,3]);write(io,1.0)
        write(io,X);write(io,y);write(io,beta);write(io,W)
      end
    end
    write(filename*".sha256",bytes2hex(sha256(read(filename)))*"  "*basename(filename)*"\n")
end
