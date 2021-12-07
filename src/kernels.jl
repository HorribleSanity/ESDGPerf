export VolumeKernel
export initialize!
export launch_volume
export rhs!

abstract type AbstractVolumeKernel end
@inline directionally_split(::AbstractVolumeKernel) = true

struct VolumeKernel{V} <: AbstractVolumeKernel end

function launch_volume(kernel::AbstractVolumeKernel, dq, q, aux, vgeo, cell, grid)
  Nq = size(cell)[1]
  Ne = length(grid)
  D = derivatives_1d(cell)[1]
  nvgeo = size(vgeo, 2)

  if directionally_split(kernel)
    @cuda threads=(Nq,Nq,Nq) blocks=Ne volume_knl(kernel, dq, q, aux, vgeo, D,
                                                  Val(Nq), Val(nvgeo), Val(1))
    @cuda threads=(Nq,Nq,Nq) blocks=Ne volume_knl(kernel, dq, q, aux, vgeo, D,
                                                  Val(Nq), Val(nvgeo), Val(2))
    @cuda threads=(Nq,Nq,Nq) blocks=Ne volume_knl(kernel, dq, q, aux, vgeo, D,
                                                  Val(Nq), Val(nvgeo), Val(3))
  else
    @cuda threads=(Nq,Nq,Nq) blocks=Ne volume_knl(kernel, dq, q, aux, vgeo, D,
                                                  Val(Nq), Val(nvgeo))
  end
end

function launch_face(dq, q, aux, vgeo, sgeo, cell, grid)
  Nq = size(cell)[1]
  Ne = length(grid)
  faceoffsets = Bennu.connectivityoffsets(cell, Val(2))
  faceix⁻, faceix⁺ = faceindices(grid)
  bndfaces = boundaryfaces(grid)

  @cuda threads=Nq^2 blocks=Ne face_knl(dq, q, aux,
                                        faceix⁻, faceix⁺, sgeo, vgeo, bndfaces,
                                        Val(Nq), Val(faceoffsets))
end

function initialize!(initf, q, aux, vgeo, t)
  Nq = Int(cbrt(size(q, 1)))
  Ne = size(q, 3)
  nvgeo = size(vgeo, 2)
  @cuda threads=Nq^3 blocks=Ne init_knl(initf, q, aux, vgeo, t, Val(Nq), Val(nvgeo))
end

function rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid)
  launch_volume(kernel, dq, q, aux, vgeo, cell, grid)
  launch_face(dq, q, aux, vgeo, sgeo, cell, grid)
end
