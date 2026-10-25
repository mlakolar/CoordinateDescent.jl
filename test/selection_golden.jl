module SelectionGoldenTests
using HighDimensionalModels, DelimitedFiles, LinearAlgebra, Random, Test
const D = get(ENV, "CD_SELECTION_GOLDEN", joinpath(@__DIR__, "data", "selection"))
readmat(name) = readdlm(joinpath(D,name*".csv"),',',Float64)
readvec(name) = vec(readmat(name))
X,y = readmat("X"),readvec("y")
@testset "R golden data" begin
  for intercept in (false,true), std in (false,true)
    a = SLOPEPath(X,y,readvec("factors");lambda=readvec("lambda"),intercept,standardize_x=std,tol=1e-9)
    b = readmat("slope-$(Int(intercept))-$(Int(std))")
    err = maximum(abs,coef(a)-b)
    println("SLOPE intercept=$intercept standardize=$std max_abs_error=$err")
    @test all(a.converged)
    @test coef(a) ≈ b atol=2e-5 rtol=1e-5
  end
  S,mu = readmat("Sigma"),readvec("mu")
  @test maximum(abs, equicorrelated_s(S) - readvec("s")) <= 2e-7
  K = gaussian_knockoffs(X,mu,S;s=readvec("s"),noise=readmat("Z"))
  println("Gaussian knockoff max_abs_error=",maximum(abs,K-readmat("Xk")))
  @test K ≈ readmat("Xk") atol=1e-8
  W = lasso_coefdiff(X,K,y,0.12;order=:cyclic,opt_tol=1e-11)
  println("Coefficient difference max_abs_error=",maximum(abs,W-readvec("W")))
  @test W ≈ readvec("W") atol=2e-6
  @test knockoff_threshold(W;q=0.3) ≈ readvec("threshold")[1] atol=2e-6
  XF = readmat("Xfixed")
  KF = fixed_knockoffs(XF;s=equicorrelated_s(XF'XF;safety=0))
  @test KF'KF ≈ readmat("fixed-gram") atol=1e-8
  @test XF'KF ≈ readmat("fixed-cross") atol=1e-8
  folds = readmat("folds")
  pairs = [(findall(>(0),folds[:,k]),findall(>(0),folds[:,k+1])) for k in 1:2:size(folds,2)]
  st = stability_selection(X,y,0.005;q=4,cutoff=0.8,pairs,intercept=false,standardize_x=false,opt_tol=1e-10)
  println("Stability frequency max_abs_error=",maximum(abs,st.frequencies-readvec("stability-frequency")))
  @test st.frequencies == readvec("stability-frequency")
  @test st.selected == Int.(readvec("stability-selected"))
  @test st.pfer_bound ≈ readvec("pfer")[1] rtol=2e-3
end
println("GOLDEN_COMPARISON_DONE")

end
