include("common.jl")
using Profile
mode=ARGS[1];rep=parse(Int,ARGS[2])
Random.seed!(8500+rep)

function timed(io,rep,case,engine,f;profile=false)
    warm=@elapsed f();GC.gc();v=nothing
    ncalls=clamp(ceil(Int,0.02/max(warm,1e-6)),1,20)
    for inner in 1:3
        r=@timed begin
            result=nothing
            for _ in 1:ncalls
                result=f()
            end
            result
        end
        v=r.value
        println(io,"$rep,$case,$engine,$inner,$(r.time/ncalls),$(r.bytes/ncalls)");flush(io)
    end
    if profile
        Profile.clear();started=time();calls=0
        Profile.@profile while calls<1 || time()-started<0.2
            f();calls+=1
        end
        open(joinpath(OUT,"profile-$rep-$case-$engine.txt"),"w") do pi
            println(pi,"rep=$rep calls=$calls Julia=$VERSION threads=$(Threads.nthreads())")
            Profile.print(pi;format=:flat,sortedby=:count,mincount=2)
        end
    end
    v
end

function benchmark(rep)
    acc=open(joinpath(OUT,"accuracy-julia-$rep.csv"),"w")
    println(acc,"rep,case,engine,max_relative_prox_residual,max_abs_difference,max_relative_objective_difference")
    open(joinpath(OUT,"timing-julia-$rep.csv"),"w") do io
        println(io,"rep,case,engine,inner,seconds,bytes")
        open(joinpath(ROOT,"inputs","bench.bin")) do bi
            while (c=readcase(bi))!==nothing
                X=c.sc==4 ? sparse(c.X) : c.X;y=c.y;λ=BH(c.p;n=c.n)
                factors=path_factors(X,y,λ)
                fs=[("CD",()->slope_cd(X,y,λ,factors)),("SLOPE.jl",()->slope_sj(X,y,λ,factors))]
                iseven(rep) && reverse!(fs)
                bs=Dict(engine=>timed(io,rep,"slope-$(c.sc)",engine,f;profile=true) for (engine,f) in fs)
                _,objref=slope_accuracy(X,y,bs["CD"],λ,factors)
                for (engine,b) in bs
                    res,obj=slope_accuracy(X,y,b,λ,factors)
                    println(acc,"$rep,slope-$(c.sc),$engine,$res,$(maximum(abs,b-bs["CD"])), $(maximum(abs.(obj-objref)./max.(1.0,abs.(objref))))")
                    writedlm(joinpath(OUT,"coef-$rep-slope-$(c.sc)-$engine.csv"),b,',')
                end
                writedlm(joinpath(OUT,"factors-$(c.sc).csv"),factors,',')
                if c.sc==1 || c.sc==2
                    S=sigma(c.p,0.3);mu=zeros(c.p)
                    # Full constructions include selecting s and covariance factorization.
                    ks=[("CD",()->CD.gaussian_knockoffs(c.X,mu,S;rng=Xoshiro(c.seed))),
                        ("Knockoffs.jl",()->KJ.modelX_gaussian_knockoffs(c.X,:equi,mu,S).Xko)]
                    iseven(rep) && reverse!(ks)
                    for (engine,f) in ks
                        timed(io,rep,"gaussian-$(c.sc)",engine,f;profile=true)
                    end
                    if c.sc==1
                        for method in (:mvr,:maxent)
                            timed(io,rep,"gaussian-$(c.sc)-$method","Knockoffs.jl",()->KJ.modelX_gaussian_knockoffs(c.X,method,mu,S);profile=true)
                        end
                        F=c.X./sqrt.(sum(abs2,c.X;dims=1))
                        fixio=open(joinpath(OUT,"fixed-validity-julia-$rep.csv"),"w")
                        println(fixio,"rep,engine,max_gram_error,max_cross_gram_error")
                        for (engine,f) in [("CD",()->CD.fixed_knockoffs(F)),("Knockoffs.jl",()->KJ.fixed_knockoffs(F,:equi))]
                            fixed=timed(io,rep,"fixed",engine,f;profile=true)
                            G=F'*F
                            Kfixed=engine=="CD" ? fixed : fixed.Xko
                            sfixed=engine=="CD" ? CD.equicorrelated_s(G) : fixed.s
                            println(fixio,"$rep,$engine,$(maximum(abs,Kfixed'*Kfixed-G)),$(maximum(abs,F'*Kfixed-(G-Diagonal(sfixed))))")
                        end
                        close(fixio)
                        K=CD.gaussian_knockoffs(c.X,mu,S;noise=c.Z)
                        open(joinpath(OUT,"bench-knockoffs.bin"),"w") do ki
                            write(ki,K)
                        end
                        A=hcat(c.X,K)
                        coeffs=Dict{String,Vector{Float64}}()
                        for (engine,f) in [("CD",()->lasso_cd(A,y,0.04)),("GLMNet.jl",()->lasso_gn(A,y,0.04))]
                            b=timed(io,rep,"joint-lasso",engine,f;profile=true);coeffs[engine]=b
                            writedlm(joinpath(OUT,"coef-$rep-lasso-$engine.csv"),b,',')
                        end
                        for (engine,f) in [("CD",()->cpss_cd(c.X,y,0.08,c.pairs,10)),("CD-threaded",()->cpss_cd(c.X,y,0.08,c.pairs,10;parallel=true)),("GLMNet.jl",()->cpss_gn(c.X,y,0.08,c.pairs,10))]
                            freq=timed(io,rep,"cpss",engine,f;profile=true)
                            writedlm(joinpath(OUT,"freq-$rep-$engine.csv"),freq,',')
                        end
                    end
                end
                flush(acc)
            end
        end
    end
    close(acc)
end

function montecarlo(rep)
    caches=Dict{Tuple{Int,Float64,Symbol},Vector{Float64}}()
    open(joinpath(OUT,"mc-julia-$rep.csv"),"w") do io
        println(io,"scenario,seed,n,p,k,method,lambda,cap,bound,selected,false,fdp,power,selected_ids")
        open(joinpath(ROOT,"inputs","shared-$rep.bin"),"w") do ko
            open(joinpath(ROOT,"inputs","mc-$rep.bin")) do bi
                while (c=readcase(bi))!==nothing
                    X,y=c.X,c.y;λ=BH(c.p;n=c.n)
                    for (engine,f) in [("CD",slope_cd),("SLOPE.jl",slope_sj)]
                        b=f(X,y,λ,[1.0])[:,1]
                        record(io,c,"slope-$engine",findall(>(1e-6),abs.(b)))
                    end
                    rho=c.sc==1 ? 0.0 : c.sc==3 ? 0.8 : 0.4
                    S=sigma(c.p,rho)
                    for method in (:equi,:mvr,:maxent)
                        s=get!(caches,(c.p,rho,method)) do
                            method==:equi ? CD.equicorrelated_s(S) : KJ.solve_s(Symmetric(S),method)
                        end
                        K=CD.gaussian_knockoffs(X,zeros(c.p),S;s,noise=c.Z)
                        write(ko,K)
                        A=hcat(X,K)
                        for (engine,f) in [("CD",lasso_cd),("GLMNet.jl",lasso_gn)]
                            W=wdifference(f(A,y,0.04))
                            record(io,c,"mx-$method-$engine",selected_w(W);lambda=0.04)
                        end
                    end
                    for method in (:equi,:mvr)
                        Random.seed!(c.seed+19)
                        K=KJ.modelX_gaussian_knockoffs(X,method,zeros(c.p),S).Xko
                        W=wdifference(lasso_gn(hcat(X,K),y,0.04))
                        record(io,c,"mx-native-$method-Knockoffs.jl",selected_w(W);lambda=0.04)
                    end
                    if c.n>=2c.p
                        K=CD.fixed_knockoffs(X);write(ko,K);A=hcat(X,K)
                        for (engine,f) in [("CD",lasso_cd),("GLMNet.jl",lasso_gn)]
                            W=wdifference(f(A,y,0.04))
                            record(io,c,"fx-$engine",selected_w(W);lambda=0.04)
                        end
                    end
                    for cap in (10,20)
                        freq=cpss_cd(X,y,0.06,c.pairs,cap)
                        record(io,c,"cpss-cap$cap-CD",findall(>=(0.8),freq);lambda=0.06,cap,bound=CD.stability_bound(c.p,cap,0.8))
                    end
                    flush(ko)
                end
            end
        end
    end
end

function validity(rep)
    # Conditional covariance check uses independent repeated draws at fixed X=0.
    # Also record deterministic covariance implied by the installed implementation.
    rng=Xoshiro(8400+rep);p=30;n=20000;S=sigma(p,0.8);X=zeros(n,p);mu=zeros(p)
    s=CD.equicorrelated_s(S);C=2Diagonal(s)-Diagonal(s)*(S\Diagonal(s))
    open(joinpath(OUT,"validity-$rep.csv"),"w") do io
        println(io,"rep,engine,relative_covariance_error,min_eigenvalue_conditional,implied_covariance_error")
        for (engine,f) in [("CD",()->CD.gaussian_knockoffs(X,mu,S;rng)),("Knockoffs.jl",()->KJ.modelX_gaussian_knockoffs(X,:equi,mu,S).Xko)]
            K=f();emp=cov(K)
            Random.seed!(8400+rep);Z=randn(n,p)
            Random.seed!(8400+rep)
            Kknown=engine=="CD" ? CD.gaussian_knockoffs(X,mu,S;s,noise=Z) : KJ.condition(X,mu,S,Diagonal(s))
            root=Z\Kknown
            implied=norm(root'*root-C)/norm(C)
            println(io,"$rep,$engine,$(norm(emp-C)/norm(C)),$(eigmin(Symmetric(C))),$implied")
        end
    end
    open(joinpath(OUT,"environment-$rep.txt"),"w") do io
        println(io,"Julia=$VERSION node=$(gethostname()) threads=$(Threads.nthreads()) BLAS=$(BLAS.get_config())")
        println(io,"SLOPE=$(pkgversion(SJ)) Knockoffs=$(pkgversion(KJ)) GLMNet=$(pkgversion(GN))")
        println(io,"Knockoffs source=$(pathof(KJ))")
    end
end

if mode=="generate"
    generate(rep)
elseif mode=="benchmark"
    benchmark(rep);validity(rep)
elseif mode=="mc"
    montecarlo(rep)
else
    error("unknown mode $mode")
end
println("DONE $mode $rep")
