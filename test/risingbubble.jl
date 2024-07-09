module RisingBubbleTests

using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using Test

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

function run(kernel, FT, N, K)
    A = CuArray
    Nq = N + 1

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
    gm = GridManager(cell, cg)
    grid = generate(meshwarp, gm)

    vgeo, sgeo = geometric_factors(kernel, grid)
    MJ = view(vgeo, :, _MJ:_MJ, :)

    ncells = length(grid)
    np = length(cell)
    q = CuArray{FT}(undef, np, nstate, ncells)
    aux = CuArray{FT}(undef, np, naux, ncells)

    initialize!(risingbubble!, q, aux, vgeo, FT(0))

    dq = similar(q)
    fill!(dq, FT(0))

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

    en0 = sqrt(sum(@. MJ * q^2))
    η0 = entropyintegral(q, aux, MJ)

    @info @sprintf """Starting
    N        = %d
    K        = %d
    kernel   = %s
    norm(q)  = %.16e
    η0       = %.16e
    """ N K kernel en0 η0

    for step = 1:numberofsteps
        for stage = 1:length(RKA)
            dq .*= RKA[stage]
            rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid)
            q .+= RKB[stage] * dt * dq
        end
    end

    enf = sqrt(sum(@. MJ * q^2))
    ηf = entropyintegral(q, aux, MJ)
    Δη = (ηf - η0) / abs(η0)

    @info @sprintf """Finished
    norm(q)      = %.16e
    ηf           = %.16e
    Δη           = %.16e
    """ enf ηf Δη

    @test abs(Δη) < 10eps(FT)

    nothing
end

function runtests()
    FT = Float64
    N = 4

    expected_error = Dict()

    K = 5
    for kernel in all_kernels
        run(kernel, FT, N, K)
    end
end

end
