export VolumeKernel
export initialize!
export launch_volume
export launch_coriolis!
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

NVTX.@annotate function launch_face(mask, dq, q, aux, vgeo, sgeo, cell, grid)
    Nq = size(cell)[1]
    Ne = length(mask)
    if Ne == 0
        return
    end

    fm = facemaps(grid)
    faceix⁻, faceix⁺ = fm.vmapM, fm.vmapP
    bndfaces = boundarycodes(grid)

    @cuda threads = Nq^2 blocks = Ne face_knl(
        mask,
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

NVTX.@annotate function launch_coriolis!(dq, q, f0, cell, grid)
    Nq = size(cell)[1]
    Ne = length(grid)
    @cuda threads = Nq^3 blocks = Ne coriolis_knl!(dq, q, f0, Val(Nq))
end

NVTX.@annotate function rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm; f_coriolis = nothing)
    NVTX.@range "Start rhs communication" begin
        start!(q, cm)
    end
    launch_volume(kernel, dq, q, aux, vgeo, cell, grid)
    launch_face(noncommunicatingcells(grid), dq, q, aux, vgeo, sgeo, cell, grid)
    NVTX.@range "Finish rhs communication" begin
        finish!(q, cm)
    end
    launch_face(communicatingcells(grid), dq, q, aux, vgeo, sgeo, cell, grid)
    if !isnothing(f_coriolis)
        launch_coriolis!(dq, q, f_coriolis, cell, grid)
    end
end

NVTX.@annotate function rkupdate!(dq, q, rka_s1, rkb_s0, dt, grid)
    Nq3 = size(q, 1)
    Ne = length(grid)
    n = Ne * Nq3 * nstate

    nthreads = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    nblocks = cld(n, nthreads)

    @cuda threads = nthreads blocks = nblocks rkupdate_knl(n, dq, q, rka_s1, rkb_s0, dt)
end
