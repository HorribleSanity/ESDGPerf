using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using WriteVTK
using Test
using MPI
using NVTX
using ArgParse
import InteractiveUtils

import ProgressMeter

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
            NVTX.@range "VTK output" begin
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
    end

    en0 = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    η0 = entropyintegral(q, aux, MJ, grid)

    if rank == 0
        @info @sprintf """Starting
        FT       = %s
        N        = %d
        K        = %d
        level    = %d
        kernel   = %s
        norm(q)  = %.16e
        η0       = %.16e
        dt       = %.16e
        nsteps   = %d
        """ FT N K level kernel en0 η0 dt numberofsteps
    end

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp; comm)

    step = 0
    time = FT(0)
    do_output(step, time, q)

    if rank == 0
        p = ProgressMeter.Progress(
            numberofsteps;
            showspeed = true,
            dt = 1,
            desc = "Time stepping...",
        )
    end
    for step = 1:numberofsteps
        NVTX.@mark "time step" payload = step

        NVTX.@range "time step" begin
            for stage = 1:length(RKA)
                NVTX.@mark "stage" payload = stage
                NVTX.@range "stage" begin
                    rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm)
                    rkupdate!(
                        dq,
                        q,
                        RKA[mod1(stage + 1, length(RKA))],
                        RKB[stage],
                        dt,
                        grid,
                    )
                end
            end
            time = dt * step
            do_output(step, time, q)
            if rank == 0
                ProgressMeter.next!(p; showvalues = [(:step, step), (:time, time)])
            end
        end
    end
    if rank == 0
        ProgressMeter.finish!(p)
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

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--enable-vtk-output"
        help = "enable vtk output"
        action = :store_true
        "--elements-per-direction", "-K"
        help = "elements per direction in coarse grid"
        arg_type = Int
        default = 1
        "--refinement-level", "-L"
        help = "refinement level passed into p4est"
        arg_type = Int
        default = 4
        "--polynomial-order", "-N"
        help = "polynomial order"
        arg_type = Int
        default = 4
        "--floating-point-type", "-F"
        help = "floating-point type (Float32 or Float64)"
        arg_type = Symbol
        default = :Float64
        "--number-of-steps", "-S"
        help = "number of time steps to take"
        arg_type = Int
        default = 50
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg, val) in parsed_args
        println("  $arg  =>  $val")
    end

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        local_comm = MPI.Comm_split_type(
            MPI.COMM_WORLD,
            MPI.COMM_TYPE_SHARED,
            MPI.Comm_rank(MPI.COMM_WORLD),
        )
        CUDA.device!(MPI.Comm_rank(local_comm) % length(devices()))
        CUDA.allowscalar(false)
    end

    if rank == 0
        InteractiveUtils.versioninfo(verbose = true)
        CUDA.versioninfo()
    end

    FT_symbol = parsed_args["floating-point-type"]

    if FT_symbol == :Float64
        FT = Float64
    elseif FT_symbol == :Float32
        FT = Float32
    else
        error("Unsupported floating-point type $(FT_symbol)")
    end

    N = parsed_args["polynomial-order"]
    K = parsed_args["elements-per-direction"]
    level = parsed_args["refinement-level"]
    fixednumberofsteps = parsed_args["number-of-steps"]
    outputvtk = parsed_args["enable-vtk-output"]

    kernel = VolumeKernel{:final}()

    run(kernel, FT, N, K, level; outputvtk, fixednumberofsteps)
end

main()
