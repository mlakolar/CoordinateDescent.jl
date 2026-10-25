# What made the old QR.jl fast

Reviewed revision `db35d869481847d10c616aa22099bcf2b3e3a203` without porting or changing QR.jl.

- [sims/helper_funcs.jl](https://github.com/mlakolar/QR.jl/blob/db35d869481847d10c616aa22099bcf2b3e3a203/sims/helper_funcs.jl) repeatedly constructs `GurobiSolver(Method=1, OutputFlag=0)`. Gurobi's [Method documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter:Method) identifies 1 as **dual simplex**, 0 as primal simplex, and 2 as barrier.
- [src/iidQR.jl](https://github.com/mlakolar/QR.jl/blob/db35d869481847d10c616aa22099bcf2b3e3a203/src/iidQR.jl) constructs a **primal LP** once: free intercept and slopes, absolute-value epigraph variables, positive/negative residuals, and residual equalities. Penalty loadings enter objective coefficients. `solve!` replaces the objective and resolves the same JuMP model. The path and post-selection refits reuse that model. No explicit transfer of coefficient starts or basis arrays appears in this source. Its design permits the LP optimizer to retain its basis; whether each historical JuMP/Gurobi version actually retained it has not been reproduced.
- The old tests instead select `MSK_OPTIMIZER_FREE_SIMPLEX`, which permits a simplex choice; that setting does not identify one fixed primal/dual variant. The Gurobi choice is unambiguous.
- QR.jl's path iterates over a supplied decreasing lambda grid. It does not implement a parametric simplex breakpoint algorithm.

## Consequence for issue78

Benchmark formulations and algorithm choices independently. In the primal LP a lambda update changes objective coefficients, preserving primal feasibility of an old basis. In the dual LP

```
maximize y'u / n
subject to 1'u = 0, tau-1 <= u <= tau,
           -n lambda w <= X'u <= n lambda w,
```

lambda updates change constraint bounds. These are different reoptimization operations. A historical preference for dual simplex in the primal formulation does not establish the fastest present solver/formulation pair.

A basis continuation approach is relevant. A fixed primal LP basis gives coefficients constant over intervals of the penalty multiplier; knots may have nonunique solutions. It is therefore incorrect to import least-squares lasso's coefficient interpolation without checking the parametrization. See [Osborne and Turlach's homotopy paper](https://maths-people.anu.edu.au/~mike/PLHty.pdf) for the distinction between continuation in the l1 constraint bound and the Lagrange multiplier. Dense lambda-grid reoptimization is a useful baseline, but it is not a full parametric simplex implementation.
