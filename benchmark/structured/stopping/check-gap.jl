# Independent dense objective/dual evaluation of the allocation-free helper.
using Random, Test, LinearAlgebra
@testset "Signal dual certificate" begin
    rng=MersenneTwister(parse(Int,ARGS[1]))
    for n in (3,8,40), alpha in (.01,1.,10.), irregular in (false,true)
        x=irregular ? cumsum(.8 .+ .4rand(rng,n)) : nothing
        K=CoordinateDescent.difference_matrix(n,1;x)
        x===nothing || (K=Diagonal(1 ./ diff(x))*K)
        D=Matrix(CoordinateDescent.difference_matrix(n-1))
        rho=.7;u=randn(rng,n-1);y=randn(rng,n);beta=randn(rng,n)
        kb=K*beta;stationarity=zeros(n);work=zeros(n-1)
        gap,objective=CoordinateDescent._stopping_signal_gap!(stationarity,work,K,beta,y,kb,u,rho,alpha)
        q=clamp.(-cumsum(rho*u)[1:end-1],-alpha,alpha)
        v=transpose(Matrix(K))*transpose(D)*q
        primal=sum(abs2,beta-y)/2+alpha*sum(abs,D*Matrix(K)*beta)
        dual=dot(y,v)-sum(abs2,v)/2
        @test all(abs.(q).<=alpha)
        @test objective≈primal rtol=2e-14
        @test gap≈primal-dual rtol=2e-14
        @test gap>=0
    end
    y=fill(3.,20);K=CoordinateDescent.difference_matrix(20)
    gap,objective=CoordinateDescent._stopping_signal_gap!(zeros(20),zeros(19),K,y,y,K*y,zeros(19),1.,1.)
    @test iszero(gap)
    @test iszero(objective)
end
