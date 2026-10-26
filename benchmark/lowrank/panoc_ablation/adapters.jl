using HighDimensionalModels,LinearAlgebra,Statistics,Profile,TOML,SHA,DelimitedFiles
import ProximalAlgorithms as PA
import ProximalOperators as PO
import ProximalCore as PC
import ProximalBase as PB
const H=HighDimensionalModels
BLAS.set_num_threads(1)
count!(::Nothing,key)=nothing
count!(s::Dict,key)=(s[key]=get(s,key,0)+1;nothing)
struct ControlledLoss{Q,D,G,S}
 d::D
 g::G
 mode::Symbol
 stats::S
 energy::Float64
end
function (f::ControlledLoss)(B)
 count!(f.stats,"value_only_calls")
 sum(abs2,f.d.Y-f.d.X*B)/(2size(f.d.X,1))
end
PC.is_generalized_quadratic(::Type{<:ControlledLoss{true}})=true
function PA.value_and_gradient(f::ControlledLoss,B)
 count!(f.stats,"value_gradient_calls")
 n=size(f.d.X,1)
 if f.mode==:legacy
  value=f(B);G=H._lr_grad(f.d,f.g,B)
 elseif f.mode==:shared
  R=f.d.X*B-f.d.Y;value=sum(abs2,R)/(2n);G=f.d.X'R/n
 else
  if f.g.A===nothing
   R=f.d.X*B-f.d.Y;value=sum(abs2,R)/(2n);G=f.d.X'R/n
  else
   G=f.g.A*B-f.g.C
   value=f.energy+(dot(B,G)-dot(B,f.g.C))/2
  end
 end
 value,G
end
struct ControlledPenalty{P,S}
 penalty::P
 stats::S
 pb::Bool
end
(g::ControlledPenalty)(B)=(count!(g.stats,"penalty_value_calls");g.penalty(B))
function PA.prox!(Y,g::ControlledPenalty,X,gamma)
 count!(g.stats,"prox_calls")
 g.pb ? PB.value_and_prox!(Y,g.penalty,X,gamma) : PC.prox!(Y,g.penalty,X,gamma)
end
function controlled(X,Y,λ,config;cached=nothing,start=nothing,stats=nothing)
 if config in ("native","apg") && cached===nothing && start===nothing
  solver=config=="native" ? :panoc : :apg
  sol=nuclear_regression(X,Y,λ;solver,max_iter=20000,tol=1e-8)
  return (;B=sol.B,iterations=sol.iterations,converged=sol.converged,solution=sol)
 end
 d=cached===nothing ? H._lr_data(X,Y,true) : cached.d
 g=cached===nothing ? H._lr_gradient(d) : cached.g
 B0=start===nothing ? zeros(eltype(d.X),size(X,2),size(Y,2)) : start
 if config in ("native","apg")
  solver=config=="native" ? :panoc : :apg
  sol=cached===nothing && start===nothing ? nuclear_regression(X,Y,λ;solver,max_iter=20000,tol=1e-8) : H._lr_nuclear(d,g,λ,B0;solver,max_iter=20000,tol=1e-8)
  return (;B=sol.B,iterations=sol.iterations,converged=sol.converged,solution=sol)
 end
 mode=config in ("gram","combined") ? :gram : config=="shared" ? :shared : :legacy
 quad=config in ("quadtrait","combined")
 f=ControlledLoss{quad,typeof(d),typeof(g),typeof(stats)}(d,g,mode,stats,sum(abs2,d.Y)/(2size(d.X,1)))
 pb=!(config in ("po95","po1_boundary"))
 penalty=pb ? PB.NuclearNorm(λ) : PO.NuclearNorm(λ)
 prox=ControlledPenalty(penalty,stats,pb)
 certpenalty=pb ? penalty : PB.NuclearNorm(λ)
 cadence=config=="cadence1" ? 1 : config=="cadence5" ? 5 : config=="cadence20" ? 20 : 10
 alpha=config=="pb99_safe" ? .99 : .95
 gamma=(config in ("pb1_boundary","po1_boundary") ? 1. : alpha)/g.L
 step=similar(B0);scale=max(1.,norm(g.C));k=Ref(0)
 function certified(B)
  count!(stats,"certificate_calls");count!(stats,"certificate_gradient_calls");G=H._lr_grad(d,g,B)
  @. step=B-G/g.L
  count!(stats,"certificate_prox_calls");PB.prox!(step,certpenalty,step,inv(g.L))
  g.L*norm(step-B)<=1e-8*scale
 end
 function stop(iter,state)
  k[]+=1;count!(stats,"callback_calls")
  k[]>1 && state.tau<1 && count!(stats,"iterations_tau_below_one")
  (k[]%cadence==0 || k[]==20000) && certified(PA.default_solution(iter,state))
 end
 B,it=PA.PANOC(;maxit=20000,stop)(;x0=copy(B0),f,g=prox,alpha,gamma,adaptive=false)
 ok=certified(B)
 if config=="coeffonly"
  return (;B,iterations=it,converged=ok,solution=nothing)
 elseif config in ("finalreuse","combined")
  E=d.Y-d.X*B;singular=svdvals(B);r=count(>(min(size(B)...)*eps(eltype(B))*first(singular)),singular)
  obj=sum(abs2,E)/(2size(X,1))+λ*sum(singular)
  sol=H._lr_solution(d,B,λ,it,ok,obj,:nuclear;coefficient_rank=r,residual=E)
 else
  obj=f(B)+prox(B);sol=H._lr_solution(d,B,λ,it,ok,obj,:nuclear)
 end
 (;B,iterations=it,converged=ok,solution=sol)
end
