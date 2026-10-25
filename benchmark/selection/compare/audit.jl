include("common.jl")
rep=parse(Int,ARGS[1])
open(joinpath(OUT,"accuracy-R-$rep.csv"),"w") do ai
    println(ai,"rep,case,engine,max_relative_prox_residual,max_abs_difference,max_relative_objective_difference")
    open(joinpath(ROOT,"inputs","bench.bin")) do bi
        while (c=readcase(bi))!==nothing
            X=c.sc==4 ? sparse(c.X) : c.X; λ=BH(c.p;n=c.n)
            factors=vec(readdlm(joinpath(OUT,"factors-$(c.sc).csv"),',',Float64))
            ref=readdlm(joinpath(OUT,"coef-$rep-slope-$(c.sc)-CD.csv"),',',Float64)
            _,oref=slope_accuracy(X,c.y,ref,λ,factors)
            for solver in ("auto","fista")
                file=joinpath(OUT,"coef-$rep-slope-$(c.sc)-R-$solver.csv")
                isfile(file) || continue
                b=readdlm(file,',',Float64)
                res,obj=slope_accuracy(X,c.y,b,λ,factors)
                diff=maximum(abs,b-ref);ogap=maximum(abs.(obj-oref)./max.(1.0,abs.(oref)))
                println(ai,"$rep,slope-$(c.sc),R-$solver,$res,$diff,$ogap")
            end
            if c.sc==1
                K=open(joinpath(OUT,"bench-knockoffs.bin")) do ki
                    read!(ki,Matrix{Float64}(undef,c.n,c.p))
                end
                A=hcat(c.X,K);λlasso=0.04
                bref=vec(readdlm(joinpath(OUT,"coef-$rep-lasso-CD.csv"),',',Float64))
                oref=sum(abs2,A*bref-c.y)/(2c.n)+λlasso*sum(abs,bref)
                for engine in ("CD","GLMNet.jl","R-glmnet")
                    b=vec(readdlm(joinpath(OUT,"coef-$rep-lasso-$engine.csv"),',',Float64))
                    g=A'*(A*b-c.y)/c.n
                    z=sign.(b-g).*max.(abs.(b-g).-λlasso,0)
                    res=norm(b-z,Inf)/max(1,norm(A'*c.y/c.n,Inf))
                    obj=sum(abs2,A*b-c.y)/(2c.n)+λlasso*sum(abs,b)
                    println(ai,"$rep,joint-lasso,$engine,$res,$(maximum(abs,b-bref)),$(abs(obj-oref)/max(1,abs(oref)))")
                end
                freqref=vec(readdlm(joinpath(OUT,"freq-$rep-CD.csv"),',',Float64))
                open(joinpath(OUT,"cpss-agreement-$rep.csv"),"w") do fi
                    println(fi,"rep,engine,max_frequency_difference")
                    for engine in ("CD-threaded","GLMNet.jl","R-stabs")
                        freq=vec(readdlm(joinpath(OUT,"freq-$rep-$engine.csv"),',',Float64))
                        println(fi,"$rep,$engine,$(maximum(abs,freq-freqref))")
                    end
                end
            end
        end
    end
end
println("AUDIT_DONE $rep")
