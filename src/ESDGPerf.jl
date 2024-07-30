module ESDGPerf
using Raven
using StaticArrays
using CUDA
using KernelAbstractions.Extras: @unroll
using MPI
using NVTX

@inline logave_ϵ(FT) = FT(1e-4)

include("contract.jl")
include("euler_gravity.jl")
include("kernels.jl")
include("geometry.jl")

include("kernels/init_kernel.jl")
include("kernels/face_kernel.jl")
include("kernels/rkupdate_kernel.jl")
include("kernels/volume_kernel_v0.jl")
include("kernels/volume_kernel_vfinal.jl")

const all_kernels = (VolumeKernel{0}(), VolumeKernel{:final}())
export all_kernels
end
