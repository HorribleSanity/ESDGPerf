module UniformTests

using ESDGPerf
using Raven
using CUDA
using StaticArrays
using Printf
using Logging
using Test
using MPI

@inline function uniform!(q, aux, _, _)
    FT = eltype(q)
    γ = _γ(FT)

    ρ = FT(1000)
    ρu⃗ = ρ .* SVector(5, 0, 10)
    p = FT(1000)
    Φ = FT(0)
    ρe = p / (γ - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ

    q[_ρ] = ρ
    q[_ρu] = ρu⃗[1]
    q[_ρv] = ρu⃗[2]
    q[_ρw] = ρu⃗[3]
    q[_ρe] = ρe

    aux[_Φ] = Φ
end

function run(kernel, FT, N, K; comm = MPI.COMM_WORLD)
    A = CuArray
    Nq = N + 1

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    vx = range(FT(0), stop = FT(20), length = K + 1)
    vy = range(FT(-5), stop = FT(5), length = K + 1)
    vz = range(FT(-5), stop = FT(5), length = 2)
    cg = brick((vx, vy, vz), (true, true, true))
    gm = GridManager(cell, cg; comm)
    grid = generate(gm)
    nlocalcells = length(grid)

    vgeo, sgeo = geometric_factors(kernel, grid)
    MJ = view(vgeo, :, _MJ:_MJ, 1:nlocalcells)

    ncells = numcells(grid, Val(true))
    np = length(cell)
    q = CuArray{FT}(undef, np, nstate, ncells)
    aux = CuArray{FT}(undef, np, naux, ncells)

    fill!(q, NaN)
    fill!(aux, NaN)

    initialize!(uniform!, q, aux, vgeo, FT(0), grid)

    cpaux = Raven.expand(nodecommpattern(grid), naux, np)
    cmaux = commmanager(eltype(q), cpaux; comm)
    share!(aux, cmaux)

    dq = similar(q)
    fill!(dq, FT(0))

    dq[:, :, (nlocalcells+1):end] .= NaN

    timeend = FT(5)
    cfl = FT(1 // 4)
    dt = cfl * min_node_distance(grid)
    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps

    RKA = (
        FT(0),
        FT(-567301805773 // 1357537059087),
        FT(-2404267990393 // 2016746695238),
        FT(-3550918686646 // 2091501179385),
        FT(-1275806237668 // 842570457699),
    )
    RKB = (
        FT(1432997174477 // 9575080441755),
        FT(5161836677717 // 13612068292357),
        FT(1720146321549 // 2090206949498),
        FT(3134564353537 // 4481467310338),
        FT(2277821191437 // 14882151754819),
    )

    en0 = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    if rank == 0
        @info @sprintf """Starting
        FT       = %s
        N        = %d
        K        = %d
        kernel   = %s
        norm(q)  = %.16e
        """ FT N K kernel en0
    end

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp; comm)

    for _ = 1:numberofsteps
        for stage in eachindex(RKA, RKB)
            dq .*= RKA[stage]
            rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
            q .+= RKB[stage] * dt * dq
        end
    end

    qexact = similar(q)
    fill!(qexact, NaN)
    initialize!(uniform!, qexact, aux, vgeo, timeend, grid)

    enf = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    errf = sqrt(
        MPI.Allreduce(
            sum(@. MJ * (q[:, :, 1:nlocalcells] - qexact[:, :, 1:nlocalcells])^2),
            +,
            comm,
        ),
    )

    if rank == 0
        @info @sprintf """Finished
        norm(q)      = %.16e
        norm(q - qe) = %.16e
        """ enf errf
    end

    errf
end

function runtests()
    if !MPI.Initialized()
        MPI.Init()
    end

    N = 4
    K = 10

    expected_error = 1.7457299184063730e-07

    for FT in (Float64,)
        for kernel in all_kernels
            err = run(kernel, FT, N, K)
            @test err ≈ expected_error atol = 10sqrt(eps(FT))
        end
    end
end

end
