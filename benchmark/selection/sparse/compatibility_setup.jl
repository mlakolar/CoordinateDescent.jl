using Pkg
root=ENV["RUN"]
Pkg.activate(joinpath(root,"env"))
Pkg.develop([PackageSpec(path=joinpath(root,"pb-source")),
             PackageSpec(path=joinpath(root,"ps-source")),
             PackageSpec(path=joinpath(root,"source"))];preserve=Pkg.PRESERVE_ALL)
Pkg.precompile()
Pkg.status(;mode=Pkg.PKGMODE_MANIFEST)
