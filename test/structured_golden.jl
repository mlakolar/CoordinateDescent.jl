module StructuredGoldenTest
using CoordinateDescent, LinearAlgebra, Test
BLAS.set_num_threads(1)
root = get(ENV, "CD_STRUCTURED_GOLDEN", joinpath(@__DIR__, "golden", "structured"))
function readmat(d,f)
  rows = [parse.(Float64, split(line)) for line in readlines(joinpath(d,f)) if !isempty(strip(line))]
  permutedims(hcat(rows...))
end
@testset "Discovery R golden paths" begin
  for seed in 8101:8103
    d = joinpath(root,string(seed))
    X, y, levels = readmat(d,"X.tsv"), vec(readmat(d,"y.tsv")), vec(readmat(d,"levels.tsv"))
    G = readmat(d,"graph_D.tsv")
    g = GeneralizedLassoPath(X,y,G,levels;abstol=1e-11,reltol=1e-10,df_tol=1e-5)
    f = fused_lasso_path(X,y,levels;l1_ratio=.25,abstol=1e-11,reltol=1e-10,df_tol=1e-5)
    for (p,tag) in ((g,"graph"),(f,"fused"))
      @test all(s->s.converged,p.solutions)
      @test coef(p) ≈ readmat(d,tag*"_beta.tsv") atol=2e-6 rtol=2e-6
      @test dof(p) == vec(readmat(d,tag*"_df.tsv"))
    end
    ys = vec(readmat(d,"signal.tsv"))
    for grid in ("unit","irregular"), order in 0:3
      x = vec(readmat(d,grid*"_x.tsv"))
      tag = grid*"_k$order"
      p = trend_filter_path(ys,levels;order,x,abstol=1e-11,reltol=1e-10,df_tol=1e-5,max_iter=50000)
      @test all(s->s.converged,p.solutions)
      ref = readmat(d,tag*"_genlasso.tsv")
      @test coef(p) ≈ ref atol=5e-6 rtol=5e-6
      @test dof(p) == vec(readmat(d,tag*"_df.tsv"))
      # glmgen uses relative objective change, so allow its looser solution
      # accuracy, while independently checking against the exact genlasso path.
      @test readmat(d,tag*"_glmgen.tsv") ≈ ref atol=2e-4 rtol=2e-4
    end
  end
end

end
