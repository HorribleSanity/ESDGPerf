using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using WriteVTK
using Test
using MPI

_grav(FT) = FT(9.81)

@inline function risingbubbleref!(state, aux, x⃗, t)
    FT = eltype(state)
    γ = _γ(FT)
    _, _, z = x⃗

    cv_d = FT(719)
    cp_d = γ * cv_d
    R_d = cp_d - cv_d

    g = _grav(FT)
    Φ = g * z

    θref = FT(300)
    p0 = FT(1e5)

    θ = θref
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

function run(
    kernel,
    FT,
    N,
    K,
    level;
    outputvtk = false,
    vtkdir = "rboutput",
    comm = MPI.COMM_WORLD,
    timeend = 100,
    fixednumberofsteps = nothing,
)
    A = CuArray
    Nq = N + 1

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    P = A(collect(toequallyspaced(cell)))

    L = FT(2e3)
    H = FT(2e3)
    vx = range(FT(-L / 2), stop = FT(L / 2), length = K + 1)
    vy = range(FT(-L / 2), stop = FT(L / 2), length = K + 1)
    vz = range(FT(0), stop = FT(H), length = K + 1)

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
    gm = GridManager(cell, cg; comm, min_level = level)
    grid = generate(gm)
    nlocalcells = length(grid)

    vgeo, sgeo = geometric_factors(kernel, grid)
    MJ = view(vgeo, :, _MJ:_MJ, 1:nlocalcells)

    ncells = numcells(grid, Val(true))
    np = length(cell)
    q = CuArray{FT}(undef, np, nstate, ncells)
    qref = CuArray{FT}(undef, np, nstate, ncells)
    aux = CuArray{FT}(undef, np, naux, ncells)

    fill!(q, NaN)
    fill!(aux, NaN)

    initialize!(risingbubbleref!, qref, aux, vgeo, FT(0), grid)
    initialize!(risingbubble!, q, aux, vgeo, FT(0), grid)

    cpaux = Raven.expand(nodecommpattern(grid), naux, np)
    cmaux = commmanager(eltype(q), cpaux; comm)
    share!(aux, cmaux)

    dq = similar(q)
    fill!(dq, FT(0))

    dq[:, :, (nlocalcells+1):end] .= NaN

    timeend = FT(timeend)
    cfl = FT(1 // 10)
    dt = cfl * min_node_distance(grid) / FT(330)
    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps
    if !isnothing(fixednumberofsteps)
        numberofsteps = fixednumberofsteps
    end

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

    outputid = "N$(N)_K$(lpad(K, 3, '0'))_level$(level)"
    if outputvtk
        rank == 0 && mkpath(vtkdir)
        pvd = rank == 0 ? paraview_collection("$(outputid)_timesteps") : nothing
    end

    do_output = function (step, time, q)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            @info "Start outputing" step time
            cd(vtkdir) do
                filename = "$(outputid)_step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)

                Pq = reshape(P * reshape(q - qref, np, :), np, nstate, :)
                vtkfile["δρ"] = Array(@view Pq[:, _ρ, 1:nlocalcells])
                vtkfile["δρu"] = Array(@view Pq[:, _ρu, 1:nlocalcells])
                vtkfile["δρv"] = Array(@view Pq[:, _ρv, 1:nlocalcells])
                vtkfile["δρw"] = Array(@view Pq[:, _ρw, 1:nlocalcells])
                vtkfile["δρe"] = Array(@view Pq[:, _ρe, 1:nlocalcells])

                vtkfile["CellNumber"] = (1:length(grid)) .+ offset(grid)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
            @info "Done outputing" step time
        end
    end


    en0 = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    η0 = entropyintegral(q, aux, MJ, grid)

    if rank == 0
        @info @sprintf """Starting
        N        = %d
        K        = %d
        level    = %d
        kernel   = %s
        norm(q)  = %.16e
        η0       = %.16e
        dt       = %.16e
        nsteps   = %d
        """ N K level kernel en0 η0 dt numberofsteps
    end

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp; comm)

    step = 0
    time = FT(0)
    do_output(step, time, q)

    for step = 1:numberofsteps
        @info "Step $(lpad(step, 6, '0'))"
        for stage = 1:length(RKA)
            rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
            rkupdate!(dq, q, RKA[mod1(stage + 1, length(RKA))], RKB[stage], dt, grid)
        end
        time = dt * step
        do_output(step, time, q)
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

    @test abs(Δη) < 100eps(FT)

    # final output
    do_output(numberofsteps, timeend, q)
    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end

    nothing
end

let
    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    end

    FT = Float64
    N = 4
    K = 1
    level = 4
    kernel = VolumeKernel{:final}()

    run(kernel, FT, N, K, level; outputvtk = false, fixednumberofsteps = 50)
end
