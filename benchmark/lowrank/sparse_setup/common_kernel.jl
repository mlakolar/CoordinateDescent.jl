# Controlled initialization experiment. The production block updates and stopping
# checks are unchanged. Centering and the RRR/SVD initialization are excluded;
# sufficient statistics and dense coefficient/result diagnostics remain timed.
function common_fit(d,A0,V0,w,λ,solver,variant,padded_L)
  A,V=copy(A0),copy(V0)
  n,p=size(d.X);q=size(d.Y,2)
  XX,XY=d.X'd.X/n,d.X'd.Y/n
  L=solver==:fista ? (variant=="candidate" ? padded_L : eigmax(Symmetric(XX))) : 0.0
  B=A*V';iteration=0;converged=false
  for k in 1:1000
    iteration=k;old=B;C=XY*V
    inner=solver==:fista ? H._lr_group_fista!(A,XX,C,λ,w;max_iter=10000,tol=1e-9,L) :
      H._lr_group!(A,XX,C,λ,w;max_iter=10000,tol=1e-9)
    S=svd(XY'A;full=false);V=S.U*S.Vt;B=A*V'
    if inner && norm(B-old)<=1e-8*max(1,norm(B))
      C=XY*V
      if H._lr_group_kkt(A,XX*A-C,λ,w)<=1e-9*max(1,norm(C))
        converged=true;break
      end
    end
  end
  E=d.Y-d.X*B
  objective=sum(abs2,E)/(2n)+λ*sum(w[j]*norm(view(A,j,:)) for j in 1:p)
  COMMON_STATE[]=(;A=copy(A),V=copy(V),w=copy(w),L)
  H._lr_solution(d,B,λ,iteration,converged,objective,:sparse;residual=E)
end
