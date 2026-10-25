# Targeted follow-up prompted by BH-weight FDR inflation in nonorthogonal designs.
# Reuses exactly the saved 840 datasets; no new timing or profiling claims.
include("common.jl")
rep=parse(Int,ARGS[1]);Random.seed!(8600+rep)
open(joinpath(OUT,"gaussian-mc-Julia-$rep.csv"),"w") do io
    println(io,"scenario,seed,n,p,k,method,lambda,cap,bound,selected,false,fdp,power,selected_ids")
    open(joinpath(ROOT,"inputs","gaussian-weights-$rep.bin"),"w") do wi
        seen=Set{Tuple{Int,Int}}()
        open(joinpath(ROOT,"inputs","mc-$rep.bin")) do bi
            while (c=readcase(bi))!==nothing
                λ=SJ.regweights(c.p;n=c.n,q=0.1,type=:gaussian)./sqrt(c.n)
                write(wi,λ)
                if !((c.n,c.p) in seen)
                    writedlm(joinpath(OUT,"gaussian-weights-$rep-$(c.n)-$(c.p).csv"),λ,',')
                    push!(seen,(c.n,c.p))
                end
                for (engine,f) in [("CD",slope_cd),("SLOPE.jl",slope_sj)]
                    b=f(c.X,c.y,λ,[1.0])[:,1]
                    record(io,c,"gaussian-$engine",findall(>(1e-6),abs.(b)))
                end
            end
        end
    end
end
println("GAUSSIAN_DONE Julia $rep")
