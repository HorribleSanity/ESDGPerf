module RisingBubbleTests

using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using Test
using MPI

_grav(FT) = FT(9.81)

@inline function risingbubble!(state, aux, x⃗, t)
    FT = eltype(state)
    γ = _γ(FT)
    x, y, z = x⃗

    cv_d = FT(719)
    cp_d = γ * cv_d
    R_d = cp_d - cv_d

    g = _grav(FT)
    Φ = g * z

    θref = FT(300)
    p0 = FT(1e5)
    rc = FT(250)
    δθc = FT(1 / 2)

    x⃗c = SVector{3,FT}(0, 0, 260)

    r = sqrt((x⃗ - x⃗c)' * (x⃗ - x⃗c))
    δθ = r <= rc ? δθc : zero(FT)

    θ = θref + δθ
    π_exner = 1 - g / (cp_d * θ) * z
    ρ = p0 / (R_d * θ) * π_exner^(cv_d / R_d)

    T = θ * π_exner
    ρe = ρ * (cv_d * T + Φ)

    state[_ρ] = ρ
    state[_ρu] = 0
    state[_ρv] = 0
    state[_ρw] = 0
    state[_ρe] = ρe

    aux[_Φ] = Φ
end

function run(kernel, FT, N, K; comm = MPI.COMM_WORLD)
    A = CuArray
    Nq = N + 1

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    L = FT(2e3)
    H = FT(2e3)
    vx = range(FT(-L / 2), stop = FT(L / 2), length = K)
    vy = range(FT(-L / 2), stop = FT(L / 2), length = K)
    vz = range(FT(0), stop = FT(H), length = K)

    function meshwarp(x)
        x₁, x₂, x₃ = x

        x₁ += L / 2
        x₂ += L / 2

        x̃₁ = x₁ + sinpi(x₁ / L) * sinpi(2 * x₂ / L) * sinpi(x₃ / H) * L / 5
        x̃₂ = x₂ - sinpi(2 * x₁ / L) * sinpi(x₂ / L) * sinpi(2 * x₃ / H) * L / 10
        x̃₃ = x₃ + sinpi(x₁ / L) * sinpi(2 * x₂ / L) * sinpi(x₃ / H) * L / 5
        SVector(x̃₁, x̃₂, x̃₃)
    end

    cg = brick((vx, vy, vz), (true, true, false))
    gm = GridManager(cell, cg; comm)
    grid = generate(meshwarp, gm)
    nlocalcells = length(grid)

    vgeo, sgeo = geometric_factors(kernel, grid)
    MJ = view(vgeo, :, _MJ:_MJ, 1:nlocalcells)

    ncells = numcells(grid, Val(true))
    np = length(cell)
    q = CuArray{FT}(undef, np, nstate, ncells)
    aux = CuArray{FT}(undef, np, naux, ncells)

    fill!(q, NaN)
    fill!(aux, NaN)

    initialize!(risingbubble!, q, aux, vgeo, FT(0), grid)

    cpaux = Raven.expand(nodecommpattern(grid), naux, np)
    cmaux = commmanager(eltype(q), cpaux; comm)
    share!(aux, cmaux)

    dq = similar(q)
    fill!(dq, FT(0))

    dq[:, :, (nlocalcells+1):end] .= NaN

    timeend = FT(100)
    cfl = FT(1 // 10)
    dt = cfl * min_node_distance(grid) / FT(330)
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
    η0 = entropyintegral(q, aux, MJ, grid)

    if rank == 0
        @info @sprintf """Starting
        N        = %d
        K        = %d
        kernel   = %s
        norm(q)  = %.16e
        η0       = %.16e
        """ N K kernel en0 η0
    end

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp; comm)

    for _ = 1:numberofsteps
        for stage = 1:length(RKA)
            rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
            rkupdate!(dq, q, RKA[mod1(stage + 1, length(RKA))], RKB[stage], dt, grid)
        end
    end

    enf = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    ηf = entropyintegral(q, aux, MJ, grid)
    Δη = (ηf - η0) / abs(η0)

    if rank == 0
        @info @sprintf """Finished
        norm(q)      = %.16e
        ηf           = %.16e
        Δη           = %.16e
        """ enf ηf Δη
    end

    @test abs(Δη) < 10eps(FT)

    nothing
end

function runtests()
    if !MPI.Initialized()
        MPI.Init()
    end

    FT = Float64
    N = 4

    expected_error = Dict()

    K = 5
    for kernel in all_kernels
        run(kernel, FT, N, K)
    end
end

end
