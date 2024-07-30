export VolumeKernel
export initialize!
export launch_volume
export rhs!
export rkupdate!
using LinearAlgebra

abstract type AbstractVolumeKernel end
@inline directionally_split(::AbstractVolumeKernel) = true

struct VolumeKernel{V} <: AbstractVolumeKernel end

NVTX.@annotate function launch_volume(
    kernel::AbstractVolumeKernel,
    dq,
    q,
    aux,
    vgeo,
    cell,
    grid,
)
    Nq = size(cell)[1]
    Ne = length(grid)
    D = derivatives_1d(cell)[1]
    nvgeo = size(vgeo, 2)

    if directionally_split(kernel)
        @cuda threads = (Nq, Nq, Nq) blocks = Ne volume_knl(
            kernel,
            dq,
            q,
            aux,
            vgeo,
            D,
            Val(Nq),
            Val(nvgeo),
            Val(1),
        )
        @cuda threads = (Nq, Nq, Nq) blocks = Ne volume_knl(
            kernel,
            dq,
            q,
            aux,
            vgeo,
            D,
            Val(Nq),
            Val(nvgeo),
            Val(2),
        )
        @cuda threads = (Nq, Nq, Nq) blocks = Ne volume_knl(
            kernel,
            dq,
            q,
            aux,
            vgeo,
            D,
            Val(Nq),
            Val(nvgeo),
            Val(3),
        )
    else
        @cuda threads = (Nq, Nq, Nq) blocks = Ne volume_knl(
            kernel,
            dq,
            q,
            aux,
            vgeo,
            D,
            Val(Nq),
            Val(nvgeo),
        )
    end
end

NVTX.@annotate function launch_face(dq, q, aux, vgeo, sgeo, cell, grid)
    Nq = size(cell)[1]
    Ne = length(grid)
    fm = facemaps(grid)
    faceix⁻, faceix⁺ = fm.vmapM, fm.vmapP
    bndfaces = boundarycodes(grid)

    @cuda threads = Nq^2 blocks = Ne face_knl(
        dq,
        q,
        aux,
        faceix⁻,
        faceix⁺,
        sgeo,
        vgeo,
        bndfaces,
        Val(Nq),
        Val(faceoffsets(cell)),
    )
end

NVTX.@annotate function initialize!(initf, q, aux, vgeo, t, grid)
    Nq = Int(cbrt(size(q, 1)))
    Ne = length(grid)
    nvgeo = size(vgeo, 2)
    @cuda threads = Nq^3 blocks = Ne init_knl(initf, q, aux, vgeo, t, Val(Nq), Val(nvgeo))
end

NVTX.@annotate function rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
    NVTX.@range "Start rhs communication" begin
        start!(q, cm)
    end
    launch_volume(kernel, dq, q, aux, vgeo, cell, grid)
    NVTX.@range "Finish rhs communication" begin
        finish!(q, cm)
    end
    launch_face(dq, q, aux, vgeo, sgeo, cell, grid)
end

NVTX.@annotate function rkupdate!(dq, q, rka_s1, rkb_s0, dt, grid)
    Nq3 = size(q, 1)
    Ne = length(grid)
    @cuda threads = Nq3 blocks = Ne rkupdate_knl(dq, q, rka_s1, rkb_s0, dt, Val(Nq3))
end
