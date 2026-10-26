using Pkg
Pkg.activate("env-baseline-1.13")
Pkg.add(PackageSpec(name="ProximalOperators",version="0.16.1");preserve=Pkg.PRESERVE_ALL)
Pkg.add(PackageSpec(name="ProximalCore",version="0.1.2");preserve=Pkg.PRESERVE_ALL)
Pkg.precompile()
using HighDimensionalModels,ProximalAlgorithms,ProximalOperators,ProximalCore,ProximalBase
Pkg.status(;mode=Pkg.PKGMODE_MANIFEST)
