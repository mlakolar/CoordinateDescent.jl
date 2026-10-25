# A one-time, analytic intercept-only basis for the compact dual LP.
# Use the documented C basis setter once; MOI basis-start attributes would
# otherwise be reapplied by the wrapper at every lambda and defeat continuation.
function nullbasis!(s,y,tau)
    opt=backend(s.m);n=length(y);nr=HiGHS.Highs_getNumRow(opt)
    @assert HiGHS.Highs_getNumCol(opt)==n
    @assert nr==2*length(s.upper)+1
    order=sortperm(y);k=clamp(ceil(Int,n*tau),1,n);pivot=order[k]
    columns=fill(HiGHS.kHighsBasisStatusUpper,n)
    columns[order[1:k-1]].=HiGHS.kHighsBasisStatusLower
    columns[pivot]=HiGHS.kHighsBasisStatusBasic
    rows=fill(HiGHS.kHighsBasisStatusBasic,nr);rows[1]=HiGHS.kHighsBasisStatusLower
    ret=HiGHS.Highs_setBasis(opt,columns,rows)
    ret==HiGHS.kHighsStatusOk || error("HiGHS rejected the analytic basis: $ret")
end
function nullpath(X,y,ls,tau,w;limit=30.)
    start=time();n,p=size(X);answers=Any[];setup=0.;optimize=0.;extract=0.;iters=0;status="complete";s=nothing
    for lambda in ls
        time()-start>=limit && (status="budget";break)
        t=time()
        if s===nothing
            s=lpmodel(X,y,lambda,tau,w,:dual,:dual;limit=max(.01,limit-(time()-start)))
            nullbasis!(s,y,tau)
        else
            lpupdate!(s,lambda,w,n)
        end
        setup+=time()-t
        set_optimizer_attribute(s.m,"time_limit",max(.01,limit-(time()-start)))
        t=time();optimize!(s.m);optimize+=time()-t
        termination_status(s.m)!=MOI.OPTIMAL && (status=string(termination_status(s.m));break)
        nit=MOI.get(backend(s.m),MOI.SimplexIterations());iters+=nit
        t=time();b,a,u=lpget(s,n);extract+=time()-t
        push!(answers,(;b,a,u,lambda,iterations=nit))
    end
    (;answers,status,setup,optimize,extract,iters,total=time()-start)
end
