export geometric_factors
export _MJ

const _g11 = 1
const _J = 10
const _MJ = 11
const _x = 12

const _sx = 1
const _sJ = 4
const _sMJ = 5

function plainarray(a::GridArray)
    a = viewwithghosts(a)

    Np = prod(size(a)[1:end-1])
    ncells = last(size(a))
    p = parent(a)

    F = Raven.fieldindex(a)
    nonfielddims = SVector((1:(F-1))..., ((F+1):ndims(p))...)
    perm = insert(nonfielddims, ndims(p) - 1, F)
    p = permutedims(p, perm)

    L = Raven.fieldslength(a)
    p = reshape(p, Np, L, ncells)

    return p
end

kernel_vgeo(::AbstractVolumeKernel, vgeo) = vgeo

function geometric_factors(kernel, grid)
    volmet = first(volumemetrics(grid))

    x⃗ = plainarray(points(grid))
    vgeo = plainarray(volmet)
    vgeo = cat(vgeo, x⃗, dims = 2)
    vgeo = kernel_vgeo(kernel, vgeo)

    surfmet = first(surfacemetrics(grid))
    sgeo = plainarray(surfmet)

    vgeo, sgeo
end
