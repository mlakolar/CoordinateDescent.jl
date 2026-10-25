using Pkg
Pkg.activate(joinpath(ENV["RUN"], "env"))
Pkg.develop([PackageSpec(path="/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot/packages/ProximalBase/86sFe"), PackageSpec(path="/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot/packages/ProximalSolvers/4JP0K"), PackageSpec(path=joinpath(ENV["RUN"], "source"))])
Pkg.add([PackageSpec(name="SLOPE",version="1.3.1"), PackageSpec(name="Knockoffs",version="3.1.4"), PackageSpec(name="GLMNet"), PackageSpec(name="DelimitedFiles"), PackageSpec(name="StatsAPI")])
Pkg.precompile()
Pkg.status(;mode=Pkg.PKGMODE_MANIFEST)
using HighDimensionalModels, SLOPE, Knockoffs, GLMNet, DelimitedFiles
println("PREPARE_DONE")
