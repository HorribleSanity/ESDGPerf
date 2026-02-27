using MPI
MPI.Init(threadlevel=:multiple)

using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using WriteVTK
using Test
using NVTX
using ArgParse
using DelimitedFiles
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
    nsaves = 100,
    vtkdir = "rboutput",
    comm = MPI.COMM_WORLD,
    timeend = 10.0,
    fixednumberofsteps = nothing,
    enabletiming = false,
    timingprefix = "timing",
    triplebuffer = nothing,
)
    A = CuArray
    Nq = N + 1

    if isnothing(triplebuffer)
        triplebuffer = Val(Raven.usetriplebuffer(A))
    end

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    P = A(collect(toequallyspaced(cell)))

    L = FT(2e3)
    H = FT(2e3)
    vx = range(FT(-L / 2), stop = FT(L / 2), length = K[1] + 1)
    vy = range(FT(-L / 2), stop = FT(L / 2), length = K[2] + 1)
    vz = range(FT(0), stop = FT(H), length = K[3] + 1)

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

    # Here we minimize the differences between Float32
    # and Float64.
    cfl = FT(0.5)
    dt = cfl * min_node_distance(grid) / 330.0
    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps
    if !isnothing(fixednumberofsteps)
        numberofsteps = fixednumberofsteps
    end
    outputstep = ceil(Int, timeend / nsaves / dt)
    dt = FT(dt)

    #RKA = (
    #    FT(0),
    #    FT(-567301805773 // 1357537059087),
    #    FT(-2404267990393 // 2016746695238),
    #    FT(-3550918686646 // 2091501179385),
    #    FT(-1275806237668 // 842570457699),
    #)
    #RKB = (
    #    FT(1432997174477 // 9575080441755),
    #    FT(5161836677717 // 13612068292357),
    #    FT(1720146321549 // 2090206949498),
    #    FT(3134564353537 // 4481467310338),
    #    FT(2277821191437 // 14882151754819),
    #)

    RKA = (
        FT(0.0),
        FT(-0.7188012108672410),
        FT(-0.7785331173421570),
        FT(-0.0053282796654044),
        FT(-0.8552979934029281),
        FT(-3.9564138245774565),
        FT(-1.5780575380587385),
        FT(-2.0837094552574054),
        FT(-0.7483334182761610),
        FT(-0.7032861106563359),
        FT(0.0013917096117681),
        FT(-0.0932075369637460),
        FT(-0.9514200470875948),
        FT(-7.1151571693922548),
    )

    RKB = (
        FT(0.0367762454319673),
        FT(0.3136296607553959),
        FT(0.1531848691869027),
        FT(0.0030097086818182),
        FT(0.3326293790646110),
        FT(0.2440251405350864),
        FT(0.3718879239592277),
        FT(0.6204126221582444),
        FT(0.1524043173028741),
        FT(0.0760894927419266),
        FT(0.0077604214040978),
        FT(0.0024647284755382),
        FT(0.0780348340049386),
        FT(5.5059777270269628),
    )

    RKC = (
        FT(0.0),
        FT(0.0367762454319673),
        FT(0.1249685262725025),
        FT(0.2446177702277698),
        FT(0.2476149531070420),
        FT(0.2969311120382472),
        FT(0.3978149645802642),
        FT(0.5270854589440328),
        FT(0.6981269994175695),
        FT(0.8190890835352128),
        FT(0.8527059887098624),
        FT(0.8604711817462826),
        FT(0.8627060376969976),
        FT(0.8734213127600976),
    )

    outputid = "N$(N)_Kx$(lpad(K[1], 2, '0'))_Ky$(lpad(K[2], 2, '0'))_Kz$(lpad(K[3], 2, '0'))_level$(level)"
    if outputvtk
        rank == 0 && mkpath(vtkdir)
        pvd = rank == 0 ? paraview_collection("$(outputid)_timesteps") : nothing
    end

    do_output = function (step, time, q)
        if outputvtk && step % outputstep == 0
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
        FT         = %s
        N          = %d
        K          = %s
        level      = %d
        kernel     = %s
        norm(q)    = %.16e
        η0         = %.16e
        dt         = %.16e
        nsteps     = %d
        outputvtk  = %s
        outputstep = %d
        timeend    = %.16e
        cuda_mpi  = %s
        """ FT N K level kernel en0 η0 dt numberofsteps outputvtk outputstep timeend MPI.has_cuda()
    end

    (begintimestep, endtimestep) = (CUDA.CuEvent(), CUDA.CuEvent())
    (prevbegintimestep, prevendtimestep) = (CUDA.CuEvent(), CUDA.CuEvent())
    timing = Vector{Float64}(undef, enabletiming ? numberofsteps : 0)

    cp = Raven.expand(nodecommpattern(grid), nstate, np)
    cm = commmanager(eltype(q), cp, triplebuffer; comm)
    if rank == 0
        @info "CommManager: $(nameof(typeof(cm)))"
    end

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
    GC.enable(false)
    for step = 1:numberofsteps
        NVTX.@mark "time step" payload = step

        NVTX.@range "time step" begin
            CUDA.record(begintimestep)
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
            CUDA.record(endtimestep)
            time = dt * step
            do_output(step, time, q)
            if rank == 0
                ProgressMeter.next!(p; showvalues = [(:step, step), (:time, time)])
            end

            if enabletiming && step > 1
                CUDA.synchronize(prevendtimestep)
                timing[step-1] = CUDA.elapsed(prevbegintimestep, prevendtimestep)
            end

            if enabletiming && step == numberofsteps
                CUDA.synchronize(endtimestep)
                timing[step] = CUDA.elapsed(begintimestep, endtimestep)
            end

            begintimestep, prevbegintimestep = prevbegintimestep, begintimestep
            endtimestep, prevendtimestep = prevendtimestep, endtimestep

            if step % 100 == 0
                GC.gc()
                MPI.Barrier(comm)
            end
        end
    end
    GC.enable(true)
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

    # turn test off for strong scaling
    #@test abs(Δη) < 100eps(FT)

    # final output
    do_output(numberofsteps, timeend, q)
    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end

    if enabletiming
        if rank == 0
            mpisize = MPI.Comm_size(comm)
            alltiming = similar(timing, numberofsteps, mpisize)
            MPI.Gatherv!(
                timing,
                VBuffer(alltiming, fill(numberofsteps, mpisize)),
                comm;
                root = 0,
            )

            alltiming = vec(alltiming)
            ranks = repeat((0:(mpisize-1)); inner = numberofsteps)
            steps = repeat((1:numberofsteps), mpisize)

            writedlm("$(timingprefix).tsv", zip(steps, ranks, alltiming))
        else
            MPI.Gatherv!(timing, nothing, comm; root = 0)
        end
    end

    nothing
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--time-end", "-T"
        help = "End time of the simulation"
        arg_type = Float64
        default = 10.0
        "--enable-vtk-output"
        help = "enable vtk output"
        action = :store_true
        "--vtk-output-directory"
        help = "directory for vtk output"
        arg_type = String
        default = "rboutput"
        "--vtk-output-number-of-saves"
        help = "Roughly the number of vtk output files"
        arg_type = Int
        default = 100
        "--elements-per-direction", "-K"
        help = "elements per direction in coarse grid"
        arg_type = Int
        default = 1
        "--elements-in-direction-x"
        help = "elements in direction x in coarse grid"
        arg_type = Int
        default = 1
        "--elements-in-direction-y"
        help = "elements in direction x in coarse grid"
        arg_type = Int
        default = 1
        "--elements-in-direction-z"
        help = "elements in direction x in coarse grid"
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
        default = -1
        "--enable-timing"
        help = "enable timing of timesteps"
        action = :store_true
        "--timing-prefix"
        help = "prefix of timing output file"
        arg_type = String
        default = "timing"
        "--enable-triple-buffer-comm"
        help = "enable triple-buffer MPI communications"
        action = :store_true
        "--enable-single-buffer-comm"
        help = "enable single-buffer MPI communications"
        action = :store_true
    end

    return parse_args(s)
end

function main()
    if !MPI.Initialized()
        MPI.Init(threadlevel=:multiple)
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    parsed_args = parse_commandline()
    if rank == 0
        println("Parsed args:")
        for (arg, val) in parsed_args
            println("  $arg  =>  $val")
        end
    end


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

    timeend = parsed_args["time-end"]
    N = parsed_args["polynomial-order"]
    K = parsed_args["elements-per-direction"]
    Kx = max(parsed_args["elements-in-direction-x"], K)
    Ky = max(parsed_args["elements-in-direction-y"], K)
    Kz = max(parsed_args["elements-in-direction-z"], K)
    level = parsed_args["refinement-level"]
    fixednumberofsteps = parsed_args["number-of-steps"]
    enabletiming = parsed_args["enable-timing"]
    timingprefix = parsed_args["timing-prefix"]
    outputvtk = parsed_args["enable-vtk-output"]
    vtkdir = parsed_args["vtk-output-directory"]
    nsaves = parsed_args["vtk-output-number-of-saves"]
    enabletriplebuffercomm = parsed_args["enable-triple-buffer-comm"]
    enablesinglebuffercomm = parsed_args["enable-single-buffer-comm"]

    if fixednumberofsteps < 0
        fixednumberofsteps = nothing
    end

    triplebuffer = if enabletriplebuffercomm
        Val(true)
    elseif enablesinglebuffercomm
        Val(false)
    else
        nothing
    end

    kernel = VolumeKernel{:final}()

    run(
        kernel,
        FT,
        N,
        (Kx, Ky, Kz),
        level;
        timeend,
        outputvtk,
        vtkdir,
        nsaves,
        fixednumberofsteps,
        enabletiming,
        timingprefix,
        triplebuffer,
    )
end

main()
