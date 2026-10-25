include("solvers.jl")
using Test, DelimitedFiles
@testset "LP formulations, signs, warm starts, and certificates" begin
    rng=Xoshiro(718)
    for tau in (.2,.5,.8), sparse in (false,true)
        X=randn(rng,80,12); sparse && (X=SparseArrays.sparse(X.*(rand(rng,size(X)...) .< .3)))
        y=X[:,1:3]*[1.,-2.,.5]+randn(rng,80); w=ones(12); lambdas=[.3,.12,.04]
        ref=lppath(X,y,lambdas,tau,w,:primal,:dual,false;limit=20.)
        @test ref.status=="complete"
        objs=[objective(X,y,s.b,s.a,s.lambda,tau,w) for s in ref.answers]
        for engine in (HAS_GUROBI ? (:highs,:gurobi) : (:highs,)), form in (:primal,:dual), strategy in (:primal,:dual,:ipm), warm in (false,true)
            path=lppath(X,y,lambdas,tau,w,form,strategy,warm;limit=20.,engine)
            @test path.status=="complete"
            @test length(path.answers)==3
            for (k,s) in enumerate(path.answers)
                c=certificate(X,y,s.b,s.a,s.lambda,tau,w,s.u)
                @test abs(c.primal-objs[k])<1e-6
                @test -1e-8<=c.gap<=1e-6
                @test c.sumdual<1e-12
            end
        end
        if !sparse
            for method in (:apg,:newton)
                path=smoothpath(X,y,lambdas,tau,w,method;h=.2,limit=30.)
                @test length(path.answers)==3
                for s in path.answers
                    kkt,u=smoothkkt(X,y,s.b,s.a,s.lambda,w,PB.ConvolutionCheck(tau,.2))
                    @test kkt<2e-5
                    c=certificate(X,y,s.b,s.a,s.lambda,tau,w,u)
                    @test c.gap>=-1e-8
                    @test c.gap<.2*3/16+1e-5
                end
            end
        end
    end
end
# One deterministic fixture for independent R acceptance checks.
rng=Xoshiro(707); X=randn(rng,150,20); X .-= mean(X,dims=1); X ./= std(X,dims=1); y=.7 .+ X[:,1:4]*[1.,-.5,.7,-1.] .+ randn(rng,150)
mkpath("golden"); writedlm("golden/X.csv",X,','); writedlm("golden/y.csv",y,',')
open("golden/julia.csv","w") do io
 println(io,"method,tau,lambda,h,objective,kkt")
 for tau in (.2,.5,.8), lambda in (.03,.1)
    path=lppath(X,y,[lambda],tau,ones(20),:primal,:dual,true)
    s=only(path.answers); c=certificate(X,y,s.b,s.a,lambda,tau,ones(20),s.u)
    println(io,"highs,$tau,$lambda,0,$(c.primal),$(c.gap)")
    writedlm("golden/exact-$(tau)-$(lambda).csv",vcat(s.a,s.b),',')
    s=only(smoothpath(X,y,[lambda],tau,ones(20),:newton;h=.2).answers)
    kkt,_=smoothkkt(X,y,s.b,s.a,lambda,ones(20),PB.ConvolutionCheck(tau,.2))
    println(io,"newton,$tau,$lambda,.2,$(objective(X,y,s.b,s.a,lambda,tau,ones(20))),$kkt")
    writedlm("golden/smooth-$(tau)-$(lambda).csv",vcat(s.a,s.b),',')
 end
end
println("VALIDATION COMPLETE")
