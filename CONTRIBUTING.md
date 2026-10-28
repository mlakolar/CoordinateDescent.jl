# Development conventions

Types and modules use UpperCamelCase. Descriptive functions, fields and keyword
arguments use snake_case; mutating functions end in `!`. Retain standard numerical
abbreviations, mathematical identifiers and names required by upstream interfaces.
Run `python3 scripts/check_naming.py` to check new declarations.

Release tests use registered dependencies and the ordinary Pkg test entry point:

```sh
julia scripts/test_registered.jl
```

This rejects path or repository overrides of runtime dependencies and runs the
suite with deprecations as errors. Unit tests use explicit kernel functions;
KernelSmoothing integration is covered by the numbered examples.

For examples or documentation, run `julia scripts/instantiate_release.jl examples`
or `julia scripts/instantiate_release.jl docs`, respectively. These environments
use this checkout and registered runtime dependencies, with KernelSmoothing
fetched at a pinned commit until it is registered.

For integration with dependency master checkouts, use a separate checkout and
run `julia scripts/test_master.jl`. Missing sibling checkouts are cloned from
master; existing checkouts are never reset or silently updated. This workflow
intentionally uses path overrides and is kept as a separate CI job.

Active manifests are generated locally. Historical benchmark lockfiles and
recorded source identities, the evidence of their original runs, are frozen in
the [benchmark snapshot](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark).
[`benchmark/README.md`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md)
describes the current benchmark harness.
HighDimensionalModels starts at 0.1.0 under a new UUID; CoordinateDescent retains
its original UUID and registered versions.
