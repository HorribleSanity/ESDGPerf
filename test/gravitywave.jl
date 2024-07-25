module GravityWaveTests

using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using Test
using MPI

_grav(FT) = FT(9.81)

@inline function gravitywave!(state, aux, x⃗, t)
    FT = eltype(state)
    γ = _γ(FT)
    x, _, z = x⃗

    g = _grav(FT)
    Φ = g * z

    cv_d = FT(719)
    cp_d = γ * cv_d
    R_d = cp_d - cv_d

    ΔT = FT(0.0001)
    H = FT(10e3)
    f = FT(0)
    L = FT(300e3)
    d = FT(5e3)
    x_c = FT(100e3)
    u_0 = FT(20)
    p_s = FT(1e5)
    T_ref = FT(250)

    δ = g / (R_d * T_ref)
    c_s = sqrt(cp_d / cv_d * R_d * T_ref)
    ρ_s = p_s / (T_ref * R_d)

    if t == 0
        δT_b = ΔT * exp(-(x - x_c)^2 / d^2) * sin(π * z / H)
        δT = exp(δ * z / 2) * δT_b
        δρ_b = -ρ_s * δT_b / T_ref
        δρ = exp(-δ * z / 2) * δρ_b
        δu, δv, δw = 0, 0, 0
    else
        xp = x - u_0 * t

        δρ_b, δu_b, δv_b, δw_b, δp_b = zeros(SVector{5,Complex{FT}})
        for m in (-1, 1)
            for n = -100:100
                k_x = 2π * n / L
                k_z = π * m / H

                p_1 = c_s^2 * (k_x^2 + k_z^2 + δ^2 / 4) + f^2
                q_1 = g * k_x^2 * (c_s^2 * δ - g) + c_s^2 * f^2 * (k_z^2 + δ^2 / 4)

                α = sqrt(p_1 / 2 - sqrt(p_1^2 / 4 - q_1))
                β = sqrt(p_1 / 2 + sqrt(p_1^2 / 4 - q_1))

                fac1 = 1 / (β^2 - α^2)
                L_m1 = (-cos(α * t) / α^2 + cos(β * t) / β^2) * fac1 + 1 / (α^2 * β^2)
                L_0 = (sin(α * t) / α - sin(β * t) / β) * fac1
                L_1 = (cos(α * t) - cos(β * t)) * fac1
                L_2 = (-α * sin(α * t) + β * sin(β * t)) * fac1
                L_3 = (-α^2 * cos(α * t) + β^2 * cos(β * t)) * fac1

                if α == 0
                    L_m1 = (β^2 * t^2 - 1 + cos(β * t)) / β^4
                    L_0 = (β * t - sin(β * t)) / β^3
                end

                δρ̃_b0 =
                    -ρ_s / T_ref * ΔT / sqrt(π) * d / L *
                    exp(-d^2 * k_x^2 / 4) *
                    exp(-im * k_x * x_c) *
                    k_z *
                    H / 2im

                δρ̃_b =
                    (
                        L_3 +
                        (p_1 + g * (im * k_z - δ / 2)) * L_1 +
                        (c_s^2 * (k_z^2 + δ^2 / 4) + g * (im * k_z - δ / 2)) * f^2 * L_m1
                    ) * δρ̃_b0

                δp̃_b = -(g - c_s^2 * (im * k_z + δ / 2)) * (L_1 + f^2 * L_m1) * g * δρ̃_b0

                δũ_b = im * k_x * (g - c_s^2 * (im * k_z + δ / 2)) * L_0 * g * δρ̃_b0 / ρ_s

                δṽ_b =
                    -f * im * k_x * (g - c_s^2 * (im * k_z + δ / 2)) * L_m1 * g * δρ̃_b0 /
                    ρ_s

                δw̃_b = -(L_2 + (f^2 + c_s^2 * k_x^2) * L_0) * g * δρ̃_b0 / ρ_s

                expfac = exp(im * (k_x * xp + k_z * z))

                δρ_b += δρ̃_b * expfac
                δp_b += δp̃_b * expfac

                δu_b += δũ_b * expfac
                δv_b += δṽ_b * expfac
                δw_b += δw̃_b * expfac
            end
        end

        δρ = exp(-δ * z / 2) * real(δρ_b)
        δp = exp(-δ * z / 2) * real(δp_b)

        δu = exp(δ * z / 2) * real(δu_b)
        δv = exp(δ * z / 2) * real(δv_b)
        δw = exp(δ * z / 2) * real(δw_b)

        δT_b = T_ref * (δp_b / p_s - δρ_b / ρ_s)
        δT = exp(δ * z / 2) * real(δT_b)
    end

    ρ_ref = ρ_s * exp(-δ * z)

    ρ = ρ_ref
    T = T_ref
    u = u_0
    w = FT(0)

    add_perturbation = true
    if add_perturbation
        ρ += δρ
        T += δT
        u += δu
        w += δw
    end

    e_kin = (u^2 + w^2) / 2
    e_int = cv_d * T
    ρe = ρ * (e_int + e_kin + Φ)

    state[_ρ] = ρ
    state[_ρu] = ρ * u
    state[_ρv] = 0
    state[_ρw] = ρ * w
    state[_ρe] = ρe

    aux[_Φ] = Φ
end

function run(kernel, FT, N, KX, KZ; comm = MPI.COMM_WORLD)
    A = CuArray
    Nq = N + 1

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    vx = range(FT(0), stop = FT(300e3), length = KX + 1)
    vy = range(-step(vx) / 2, stop = step(vx) / 2, length = 2)
    vz = range(FT(0), stop = FT(10e3), length = KZ + 1)
    cg = brick((vx, vy, vz), (true, true, false))
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

    initialize!(gravitywave!, q, aux, vgeo, FT(0), grid)

    cpaux = Raven.expand(nodecommpattern(grid), naux, np)
    cmaux = commmanager(eltype(q), cpaux; comm)
    share!(aux, cmaux)

    dq = similar(q)
    fill!(dq, FT(0))

    dq[:, :, (nlocalcells+1):end] .= NaN

    timeend = FT(10 * 60)
    cfl = FT(1 // 4)
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
    if rank == 0
        @info @sprintf """Starting
        N        = %d
        KX       = %d
        KZ       = %d
        kernel   = %s
        norm(q)  = %.16e
        """ N KX KZ kernel en0
    end

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp; comm)

    for _ = 1:numberofsteps
        for stage in eachindex(RKA, RKB)
            rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
            rkupdate!(dq, q, RKA[mod1(stage + 1, length(RKA))], RKB[stage], dt, grid)
        end
    end

    qexact = similar(q)
    fill!(qexact, NaN)
    initialize!(gravitywave!, qexact, aux, vgeo, timeend, grid)

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

    FT = Float64
    N = 4
    nlevels = 2

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    expected_error = Dict()
    expected_error[1] = 7.4700628972712752e+02
    expected_error[2] = 1.4229051094831473e+02

    for kernel in all_kernels
        errs = zeros(FT, nlevels)
        for l = 1:nlevels
            KX = 30 * 2^(l - 1)
            KZ = 3 * 2^(l - 1)
            errs[l] = run(kernel, FT, N, KX, KZ; comm)
            @test errs[l] ≈ expected_error[l] rtol = 20sqrt(eps(FT))
        end
        if nlevels > 1
            rates = log2.(errs[1:(nlevels-1)] ./ errs[2:nlevels])
            if rank == 0
                @info "Convergence rates\n" * join(
                    ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(nlevels-1)],
                    "\n",
                )
            end
            @test rates[1] ≈ 2.39 atol = 0.01
        end
    end
end

end
