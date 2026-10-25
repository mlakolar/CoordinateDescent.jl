# Descriptive comparison of already-completed R fits. Does not refit a model.
include("solvers.jl")
function loadbinary(file)
    open(file) do io
        read!(io,Vector{Float64}(undef,filesize(file)÷8))
    end
end
# Epanechnikov convolution conjugate: if q=u-(tau-1/2),
# s=2sin(asin(2q)/3), then phi_h*(u)=-3h(1-s^2)^2/16.
function smoothconjugate(u,tau,h)
    s=2sin(asin(clamp(2*(u-tau+.5),-1.,1.))/3)
    -3h*(1-s^2)^2/16
end
function feasibledual(X,lambda,u)
    lo,hi=minimum(u)-.5,maximum(u)+.5
    for _ in 1:65
        c=(lo+hi)/2
        sum(clamp.(u.-c,-.5,.5))>0 ? (lo=c) : (hi=c)
    end
    v=clamp.(u.-(lo+hi)/2,-.5,.5)
    v./max(1.,maximum(abs.(X'*v))./(size(X,1)*lambda)*(1+1e-12))
end
function main()
 root=ARGS[1];out=ARGS[2];mkpath(dirname(out))
 for tau in (.2,.5,.8),h in (.05,.2),r in range(-1.,1.,length=201)
    f,d=PB.value_and_deriv(PB.ConvolutionCheck(tau,h),0.,r);u=-d
    @assert abs(f+smoothconjugate(u,tau,h)-u*r)<1e-12
 end
 open(out,"w") do io
  println(io,"case,seed,rep,k,lambda,exact_objective,smoothed_check_objective,check_excess,coefficient_l2,training_prediction_rms,smooth_objective_gain,smoothing_bound,smoothed_primal,smoothed_dual,smoothed_dual_gap")
  for case in ("small","tall","tall100k"),seed in (701,702,703)
   n=Dict("small"=>1000,"tall"=>10000,"tall100k"=>100000)[case];p=100
   base=joinpath(root,"$case-$seed")
   all(isfile(base*suffix) for suffix in ("-X.bin","-y.bin","-lambda.bin","-quantreg-fn-1.bin","-quantreg-fn-2.bin","-conquer-1.bin","-conquer-2.bin")) || continue
   X=reshape(loadbinary(base*"-X.bin"),n,p);y=loadbinary(base*"-y.bin");ls=loadbinary(base*"-lambda.bin")
   loss=PB.ConvolutionCheck(.5,.2);w=ones(p)
   for rep in 1:2
    exact=reshape(loadbinary(base*"-quantreg-fn-$rep.bin"),p+1,8)
    smooth=reshape(loadbinary(base*"-conquer-$rep.bin"),p+1,8)
    for k in eachindex(ls)
     be=exact[2:end,k];ae=exact[1,k];bs=smooth[2:end,k];as=smooth[1,k];lambda=ls[k]
     pe=objective(X,y,be,ae,lambda,.5,w);ps=objective(X,y,bs,as,lambda,.5,w)
     fe=first(smoothfg(X,y,be,ae,loss))+lambda*penalty(be,w)
     fs=first(smoothfg(X,y,bs,as,loss))+lambda*penalty(bs,w)
     _,_,_,us=smoothfg(X,y,bs,as,loss)
     v=feasibledual(X,lambda,us)
     ds=dot(y,v)/n-mean(smoothconjugate.(v,.5,.2))
     @assert fs-ds>=-1e-7
     difference=X*(bs-be) .+ (as-ae)
     println(io,join((case,seed,rep,k,lambda,pe,ps,ps-pe,norm(bs-be),sqrt(mean(abs2,difference)),fe-fs,3*.2/16,fs,ds,fs-ds),','))
    end
   end
  end
 end
end
main()
