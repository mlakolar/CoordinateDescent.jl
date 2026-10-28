using Serialization, Test
root=joinpath(@__DIR__,"results")
cases=("fused40","fused100","fused300","graph100","trend40","trend100","rankdef100","wide100")
maxcoef,maxobj,points=0.,0.,0
@testset "paired unchanged estimator and diagnostics" begin
    for rep in 1:7, label in cases
        a=deserialize(joinpath(root,"fit-baseline-$rep-$label.bin"))
        b=deserialize(joinpath(root,"fit-candidate-$rep-$label.bin"))
        @test a.inputdigest == b.inputdigest
        @test a.x == b.x
        @test a.objective == b.objective
        @test a.df == b.df
        @test a.iterations == b.iterations
        @test all(a.converged) && all(b.converged)
        global maxcoef=max(maxcoef,maximum(abs,a.x-b.x))
        global maxobj=max(maxobj,maximum(abs,a.objective-b.objective))
        global points+=length(a.df)
    end
end
open(joinpath(root,"validation.txt"),"w") do io
    println(io,"datasets=7 cases=8 path_points=$points max_coefficient_difference=$maxcoef max_objective_difference=$maxobj exact_df=true exact_iterations=true all_converged=true")
end
