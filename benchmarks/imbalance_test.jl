using MPI
using Raven
using ArgParse

MPI.Init()

function build_grid(
    FT,
    N,
    K,
    level;
    comm = MPI.COMM_WORLD,
)
    Nq = N + 1
    cell = LobattoCell{FT,Array}(Nq, Nq, Nq)

    # Rising bubble params
    domain_width = FT(2e3)
    domain_height = FT(2e3)
    vx = range(-domain_width / 2; stop = domain_width / 2, length = K[1] + 1)
    vy = range(-domain_width / 2; stop = domain_width / 2, length = K[2] + 1)
    vz = range(zero(FT); stop = domain_height, length = K[3] + 1)

    coarse_grid = brick((vx, vy, vz), (true, true, false))
    grid_manager = GridManager(cell, coarse_grid; comm, min_level = level)
    grid = generate(grid_manager)

    pattern = nodecommpattern(grid)
    rank = MPI.Comm_rank(comm)

    metrics = (
        cells = length(grid),
        communicating_cells = length(communicatingcells(grid)),
        send_nodes = length(pattern.sendindices),
        neighbor_ranks = length(pattern.sendranks),
    )

    all_metrics = MPI.gather(metrics, comm; root = 0)

    if rank == 0
        for name in propertynames(metrics)
            values = getproperty.(all_metrics, name)
            mean_value = sum(values) / length(values)
            imbalance = maximum(values) / mean_value

            println(
                "$name: min=$(minimum(values)), mean=$mean_value, ",
                "max=$(maximum(values)), max/mean=$imbalance",
            )
        end
    end

    return grid
end

function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--elements-per-direction", "-K"
        help = "elements per direction in the coarse grid"
        arg_type = Int
        default = 1
        "--elements-in-direction-x"
        help = "coarse-grid elements in the x direction"
        arg_type = Int
        default = 1
        "--elements-in-direction-y"
        help = "coarse-grid elements in the y direction"
        arg_type = Int
        default = 1
        "--elements-in-direction-z"
        help = "coarse-grid elements in the z direction"
        arg_type = Int
        default = 1
        "--refinement-level", "-L"
        help = "refinement level passed to p4est"
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
    end

    return parse_args(settings)
end

function main()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    parsed_args = parse_commandline()

    if rank == 0
        println("Parsed args:")
        for (arg, value) in parsed_args
            println("  $arg => $value")
        end
    end

    FT_symbol = parsed_args["floating-point-type"]
    FT = if FT_symbol == :Float64
        Float64
    elseif FT_symbol == :Float32
        Float32
    else
        error("Unsupported floating-point type $FT_symbol")
    end

    N = parsed_args["polynomial-order"]
    K = parsed_args["elements-per-direction"]
    Kx = max(parsed_args["elements-in-direction-x"], K)
    Ky = max(parsed_args["elements-in-direction-y"], K)
    Kz = max(parsed_args["elements-in-direction-z"], K)
    level = parsed_args["refinement-level"]

    build_grid(FT, N, (Kx, Ky, Kz), level; comm)
end

main()
MPI.Finalize()
