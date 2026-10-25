using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
