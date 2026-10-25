# Generate shared numerical inputs, or compare the independent R fits.
using CoordinateDescent, LinearAlgebra, Random, DelimitedFiles, Test, Printf
mode, out = ARGS
mkpath(out)
if mode == "generate"
  for case in 1:3
    rng = Xoshiro(8100+case)
    n,p,q,r = case == 1 ? (100,10,6,2) : case == 2 ? (80,12,8,2) : (30,40,5,2)
    X = randn(rng,n,p)
    U = zeros(p,r); U[1:4,:] = randn(rng,4,r)
    V = randn(rng,q,r)
    Y = X*U*V' + 0.2randn(rng,n,q)
    writedlm(joinpath(out,"X$case.csv"),X,',')
    writedlm(joinpath(out,"Y$case.csv"),Y,',')
  end
elseif mode == "compare"
  @testset "rrpack golden comparisons" begin
    for case in 1:3
      X = readdlm(joinpath(out,"X$case.csv"),',',Float64)
      Y = readdlm(joinpath(out,"Y$case.csv"),',',Float64)
      df = vec(readdlm(joinpath(out,"df-$case.csv"),',',Float64))
      for r in (1,2,3)
        s = reduced_rank(X,Y,r;intercept=false)
        ref = readdlm(joinpath(out,"rrr-$case-$r.csv"),',',Float64)
        wide = size(X,2) > size(X,1)
        err = maximum(abs, wide ? X*(s.B-ref) : s.B-ref)
        @printf("rrr case=%d rank=%d %s_error=%.6g\n",case,r,wide ? "fitted" : "coefficient",err)
        if size(X,2) > size(X,1)
          # rrpack's pivoted QR returns a basic solution; our SVD returns
          # minimum-norm coefficients. The identifiable fitted values agree.
          @test X*s.B ≈ X*ref atol=2e-9 rtol=2e-9
          @test norm(s.B) <= norm(ref) + 2e-9
        else
          @test s.B ≈ ref atol=2e-9 rtol=2e-9
        end
        @test s.df_exact ≈ df[r] atol=2e-8 rtol=2e-8
      end
      case == 3 && continue # R's nonconvex fits are checked on identifiable designs.
      for λ in (0.01,0.05,0.2), adaptive in (false,true)
        s = sparse_reduced_rank(X,Y,2,λ;intercept=false,adaptive,tol=1e-10,inner_tol=1e-11)
        tag = "srrr-$case-$λ-$adaptive"
        ref = readdlm(joinpath(out,"$tag.csv"),',',Float64)
        err = maximum(abs,s.B-ref)
        @printf("%s maxerror=%.6g converged=%s\n",tag,err,s.converged)
        @test s.converged
        @test s.B ≈ ref atol=1e-7 rtol=1e-7
      end
      ss = sparse_svd_regression(X,Y,2;intercept=false,nλ=10,tol=1e-10,inner_tol=1e-11)
      ref = readdlm(joinpath(out,"rssvd-$case.csv"),',',Float64)
      err = maximum(abs,ss.solution.B-ref)
      @printf("rssvd case=%d maxerror=%.6g\n",case,err)
      @test ss.solution.B ≈ ref atol=1e-7 rtol=1e-7
    end
  end
else
  error("expected generate or compare")
end
