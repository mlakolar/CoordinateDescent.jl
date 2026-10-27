# Explicit kernel functions keep unit tests independent of KernelSmoothing.
# Integration with that package is exercised by the numbered examples.
module TestKernels
export EpanechnikovKernel, GaussianKernel, UniformKernel
EpanechnikovKernel() = u -> abs(u) <= 1 ? 0.75 * (1 - u^2) : 0.0
GaussianKernel() = u -> exp(-u^2 / 2) / sqrt(2pi)
UniformKernel() = u -> abs(u) <= 1 ? 0.5 : 0.0
end
