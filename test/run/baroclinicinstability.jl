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

_grav(FT) = FT(9.80616)
_omega(FT) = FT(7.29212e-5)

# Compute geopotential and temperature at a given eta-level (pressure coordinate).
# Translated from Fortran subroutine baroclinic_instability_functions
# (initial_conditions.F90, lines 6343-6376).
# Uses f-plane approximation (beta0 = 0).
# Look at Ulrich et al eq (8)
@inline function baroclinic_functions(y, eta, f0, u0, T0, gamma_lapse, b, Ly, R_d, g)
    b2 = b * b
    twopi = 2 * oftype(y, pi)

    # Mean temperature and geopotential (hydrostatic reference)
    exp1 = R_d * gamma_lapse / g
    Tmean = T0 * eta^exp1
    phimean = T0 * g / gamma_lapse * (1 - eta^exp1)

    logeta = log(eta)

    # Meridional structure (beta0 = 0, so fac2 term vanishes)
    fac1 = f0 * (y - Ly / 2 - Ly / twopi * sin(twopi * y / Ly))
    fac3 = exp(-logeta * logeta / b2)
    phi_prime = u0 / 2 * fac1

    geo_phi = phimean + phi_prime * fac3 * logeta
    temp = Tmean + phi_prime / R_d * fac3 * (2 / b2 * logeta * logeta - 1)

    return (geo_phi, temp)
end

# Reference state: balanced jet without perturbation.
# Translated from Fortran subroutine baroclinic_instability_cube
# (initial_conditions.F90, lines 6216-6338), with iperturbation = 0.
@inline function baroclinicref!(state, aux, x⃗, t)
    FT = eltype(state)
    γ = _γ(FT)
    _, y, z = x⃗

    cv_d = FT(717.5)
    cp_d = γ * cv_d
    R_d = cp_d - cv_d

    g = _grav(FT)
    Φ = g * z

    p0 = FT(1e5)

    # Baroclinic instability parameters (from Fortran code)
    u0 = FT(35)
    T0 = FT(288)
    gamma_lapse = FT(0.005)
    b = FT(2)
    b2 = b * b
    Ly = FT(6e6)
    f0 = 2 * _omega(FT) / sqrt(FT(2))

    # Newton iteration to find eta (= p/p0) from geometric height z.
    # Solves: geo_phi(eta) = g * z
    eta = FT(1e-14)
    local temp
    for _ = 1:100
        geo_phi, temp = baroclinic_functions(y, eta, f0, u0, T0, gamma_lapse, b, Ly, R_d, g)
        num = -g * z + geo_phi
        den = -R_d / eta * temp
        deta = num / den
        eta = eta - deta
        abs(deta) <= FT(1e-14) && break
    end
    eta = clamp(eta, FT(0), FT(1))

    # Thermodynamic state
    T = temp
    press = p0 * eta
    ρ = press / (R_d * T)

    # Balanced zonal wind
    logeta = log(eta)
    u = -u0 * sin(FT(pi) * y / Ly)^2 * logeta * exp(-logeta * logeta / b2)

    ρe = ρ * (cv_d * T + u^2 / 2 + Φ)

    state[_ρ] = ρ
    state[_ρu] = ρ * u
    state[_ρv] = 0
    state[_ρw] = 0
    state[_ρe] = ρe

    aux[_Φ] = Φ
end

# Perturbed state: balanced jet + Gaussian velocity perturbation.
# Translated from Fortran subroutine baroclinic_instability_cube
# (initial_conditions.F90, lines 6216-6338), with iperturbation = 1.
@inline function baroclinicinstability!(state, aux, x⃗, t)
    FT = eltype(state)
    γ = _γ(FT)
    x, y, z = x⃗

    cv_d = FT(717.5)
    cp_d = γ * cv_d
    R_d = cp_d - cv_d

    g = _grav(FT)
    Φ = g * z

    p0 = FT(1e5)

    # Baroclinic instability parameters (from Fortran code)
    u0 = FT(35)
    T0 = FT(288)
    gamma_lapse = FT(0.005)
    b = FT(2)
    b2 = b * b
    Ly = FT(6e6)
    f0 = 2 * _omega(FT) / sqrt(FT(2))

    # Perturbation parameters (from Fortran code)
    up = FT(1)
    Lp = FT(6e5)
    xc = FT(2e6)
    yc = FT(2.5e6)

    # Newton iteration to find eta (= p/p0) from geometric height z.
    eta = FT(1e-14)
    local temp
    for _ = 1:100
        geo_phi, temp = baroclinic_functions(y, eta, f0, u0, T0, gamma_lapse, b, Ly, R_d, g)
        num = -g * z + geo_phi
        den = -R_d / eta * temp
        deta = num / den
        eta = eta - deta
        abs(deta) <= FT(1e-14) && break
    end
    eta = clamp(eta, FT(0), FT(1))

    # Thermodynamic state
    T = temp
    press = p0 * eta
    ρ = press / (R_d * T)

    # Balanced zonal wind
    logeta = log(eta)
    u = -u0 * sin(FT(pi) * y / Ly)^2 * logeta * exp(-logeta * logeta / b2)

    # Gaussian perturbation to trigger instability
    rc2 = (x - xc)^2 + (y - yc)^2
    du = up * exp(-rc2 / (Lp * Lp))

    ρe = ρ * (cv_d * T + (u + du)^2 / 2 + Φ)

    state[_ρ] = ρ
    state[_ρu] = ρ * (u + du)
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
    vtkdir = "bcoutput",
    comm = MPI.COMM_WORLD,
    timeend = 864000.0,
    fixednumberofsteps = nothing,
    enabletiming = false,
    timingprefix = "timing",
    triplebuffer = nothing,
    timestep = 2.0,
)
    A = CuArray
    Nq = N + 1

    if isnothing(triplebuffer)
        triplebuffer = Val(Raven.usetriplebuffer(A))
    end

    rank = MPI.Comm_rank(comm)

    cell = LobattoCell{FT,A}(Nq, Nq, Nq)

    P = A(collect(toequallyspaced(cell)))

    # Domain: Lx=40000km (periodic), Ly=6000km (walled), H=30km (walled)
    Lx = FT(4e7)
    Ly = FT(6e6)
    H = FT(3e4)
    vx = range(FT(0), stop = Lx, length = K[1] + 1)
    vy = range(FT(0), stop = Ly, length = K[2] + 1)
    vz = range(FT(0), stop = H, length = K[3] + 1)

    # Periodic in x, walled in y and z (channel configuration)
    cg = brick((vx, vy, vz), (true, false, false))
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

    initialize!(baroclinicref!, qref, aux, vgeo, FT(0), grid)
    initialize!(baroclinicinstability!, q, aux, vgeo, FT(0), grid)

    cpaux = Raven.expand(nodecommpattern(grid), naux, np)
    cmaux = commmanager(eltype(q), cpaux; comm)
    share!(aux, cmaux)

    dq = similar(q)
    fill!(dq, FT(0))

    dq[:, :, (nlocalcells+1):end] .= NaN

    # Coriolis parameter for f-plane (from Fortran: 2*omega/sqrt(2))
    f0 = 2 * _omega(FT) / sqrt(FT(2))

    # Sponge layer (Klemp et al. MWR 2008, NUMA ugwal_sponge_weight)
    g = _grav(FT)
    sponge_thickness = FT(7000)
    z_sponge_start = H - sponge_thickness
    α_sponge_max = FT(0.4)
    sponge_params = (α_sponge_max, z_sponge_start, H, g)

    # Sound speed at T0=288K is ~340 m/s
    # cfl = FT(0.25)
    # dt_cfl = cfl * min_node_distance(grid) / 340.0
    # The paper runs instead use a fixed timestep supplied on the command line.
    dt_cfl = FT(timestep)
    # Reduce dt slightly so it divides evenly into half a day (43200s)
    steps_per_halfday = ceil(Int, 43200 / dt_cfl)
    dt = FT(43200.0 / steps_per_halfday)
    numberofsteps = round(Int, timeend / 43200) * steps_per_halfday
    if !isnothing(fixednumberofsteps)
        numberofsteps = fixednumberofsteps
    end
    outputstep = 2 * steps_per_halfday  # output every day (was every half day)
    # numberofsteps = ceil(Int, timeend / dt)
    # dt = timeend / numberofsteps
    # outputstep = ceil(Int, timeend / nsaves / dt)
    # dt = FT(dt)

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
        # Ad or condition here
        # (step * dt == 86400 * 12)
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

                    # Derived fields: pressure and temperature
                    Pq_full = reshape(P * reshape(q, np, :), np, nstate, :)
                    Paux_full = reshape(P * reshape(aux, np, :), np, naux, :)

                    ρ_a = Array(@view Pq_full[:, _ρ, 1:nlocalcells])
                    ρu_a = Array(@view Pq_full[:, _ρu, 1:nlocalcells])
                    ρv_a = Array(@view Pq_full[:, _ρv, 1:nlocalcells])
                    ρw_a = Array(@view Pq_full[:, _ρw, 1:nlocalcells])
                    ρe_a = Array(@view Pq_full[:, _ρe, 1:nlocalcells])
                    Φ_a = Array(@view Paux_full[:, _Φ, 1:nlocalcells])

                    γ_val = _γ(FT)
                    R_d_val = (γ_val - 1) * FT(717.5)
                    pressure = @. (γ_val - 1) * (ρe_a - (ρu_a^2 + ρv_a^2 + ρw_a^2) / (2 * ρ_a) - ρ_a * Φ_a)
                    temperature = @. pressure / (ρ_a * R_d_val)

                    vtkfile["ρ"] = ρ_a
                    vtkfile["ρu"] = ρu_a
                    vtkfile["ρv"] = ρv_a
                    vtkfile["ρw"] = ρw_a
                    vtkfile["ρe"] = ρe_a
                    vtkfile["pressure"] = pressure
                    vtkfile["temperature"] = temperature

                    vtkfile["CellNumber"] = (1:length(grid)) .+ offset(grid)
                    vtk_save(vtkfile)
                    if rank == 0
                        pvd[time] = vtkfile
                    end
                end
                GC.gc()
                CUDA.reclaim()
                @info "Done outputing" step time
            end
        end
        if step % outputstep == 0
            local_m = sum(@. MJ * q[:, _ρ:_ρ, 1:nlocalcells])
            local_e = sum(@. MJ * q[:, _ρe:_ρe, 1:nlocalcells])
            total_m = MPI.Allreduce(local_m, +, comm)
            total_e = MPI.Allreduce(local_e, +, comm)
            if rank == 0
                @info @sprintf("Conservation step=%d time=%.16e mass=%.16e energy=%.16e", step, time, total_m, total_e)
            end
        end
    end

    en0 = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    η0 = entropyintegral(q, aux, MJ, grid)
    m0 = MPI.Allreduce(sum(@. MJ * q[:, _ρ:_ρ, 1:nlocalcells]), +, comm)
    e0 = MPI.Allreduce(sum(@. MJ * q[:, _ρe:_ρe, 1:nlocalcells]), +, comm)

    if rank == 0
        @info @sprintf """Starting
        FT         = %s
        N          = %d
        K          = %s
        level      = %d
        kernel     = %s
        norm(q)    = %.16e
        η0         = %.16e
        m0         = %.16e
        e0         = %.16e
        dt         = %.16e
        nsteps     = %d
        outputvtk  = %s
        outputstep = %d
        timeend    = %.16e
        cuda_mpi  = %s
        """ FT N K level kernel en0 η0 m0 e0 dt numberofsteps outputvtk outputstep timeend MPI.has_cuda()
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
    for step = 1:numberofsteps
        NVTX.@mark "time step" payload = step

        NVTX.@range "time step" begin
            CUDA.record(begintimestep)
            for stage = 1:length(RKA)
                NVTX.@mark "stage" payload = stage
                NVTX.@range "stage" begin
                    rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid, cm; f_coriolis = f0, sponge = nothing)
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

        end
    end
    if rank == 0
        ProgressMeter.finish!(p)
    end

    enf = sqrt(MPI.Allreduce(sum(@. MJ * q[:, :, 1:nlocalcells]^2), +, comm))
    ηf = entropyintegral(q, aux, MJ, grid)
    Δη = (ηf - η0) / abs(η0)
    mf = MPI.Allreduce(sum(@. MJ * q[:, _ρ:_ρ, 1:nlocalcells]), +, comm)
    ef = MPI.Allreduce(sum(@. MJ * q[:, _ρe:_ρe, 1:nlocalcells]), +, comm)
    Δm = (mf - m0) / abs(m0)
    Δe = (ef - e0) / abs(e0)

    if rank == 0
        @info @sprintf """Finished
        norm(q)      = %.16e
        ηf           = %.16e
        Δη           = %.16e
        mf           = %.16e
        ef           = %.16e
        Δm           = %.16e
        Δe           = %.16e
        """ enf ηf Δη mf ef Δm Δe
    end

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
        default = 864000.0
        "--time-step"
        help = "time step in seconds"
        arg_type = Float64
        default = 2.0
        "--enable-vtk-output"
        help = "enable vtk output"
        action = :store_true
        "--vtk-output-directory"
        help = "directory for vtk output"
        arg_type = String
        default = "bcoutput"
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
        help = "elements in direction y in coarse grid"
        arg_type = Int
        default = 1
        "--elements-in-direction-z"
        help = "elements in direction z in coarse grid"
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
        InteractiveUtils.versioninfo()
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
    timestep = parsed_args["time-step"]
    timestep > 0 || error("Time step must be positive")
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
        timestep,
    )
end

main()
