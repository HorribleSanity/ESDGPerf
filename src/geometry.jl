export geometric_factors
export _MJ

const _g11 = 1
const _J = 10
const _MJ = 11
const _x = 12

using StructArrays: StructArray

rcomponents(a) = a
rcomponents(a::StructArray) = map(rcomponents, components(a))
rcat(a::AbstractArray) = reshape(a, size(a, 1), 1, size(a, 2))
rcat(a) = cat(map(rcat, a)..., dims= 2)
plainarray(a::StructArray) = rcat(rcomponents(a))

kernel_vgeo(::AbstractVolumeKernel, vgeo) = vgeo

function geometric_factors(kernel, cell, grid)
  met = metrics(grid)
  _, J = components(met)
  M = mass(cell)
  MJ = M * J
  MJ = reshape(MJ, size(MJ, 1), 1, size(MJ, 2))

  x⃗ = plainarray(points(grid))
  vgeo = plainarray(met)
  vgeo = cat(vgeo, MJ, x⃗, dims=2)
  vgeo = kernel_vgeo(kernel, vgeo)

  facemet = facemetrics(grid)
  faceM = facemass(cell)
  _, faceJ = components(facemetrics(grid))
  faceMJ = faceM * faceJ
  faceMJ = reshape(faceMJ, size(faceMJ, 1), 1, size(faceMJ, 2))

  sgeo = plainarray(facemet)
  sgeo = cat(sgeo, faceMJ, dims=2)

  vgeo, sgeo
end
