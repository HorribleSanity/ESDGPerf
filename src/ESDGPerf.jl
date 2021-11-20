module ESDGPerf
  using Reexport
  @reexport using Bennu
  using StaticArrays
  using CUDA
  using KernelAbstractions.Extras: @unroll

  include("euler_gravity.jl")
  include("kernels.jl")
  include("geometry.jl")

  include("kernels/init_kernel.jl")
  include("kernels/face_kernel.jl")
  include("kernels/volume_kernel_v0.jl")
  include("kernels/volume_kernel_v1.jl")
end
