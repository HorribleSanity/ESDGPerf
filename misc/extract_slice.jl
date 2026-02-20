# julia --project=misc/slice_env misc/extract_slice.jl <vtk_dir> [z=937.5] [outdir=slices]
using ReadVTK, WriteVTK, DelimitedFiles

vtkdir, z_target = ARGS[1], length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 937.5
outdir = length(ARGS) >= 3 ? ARGS[3] : "slices"; mkpath(outdir)

stepnum(d) = parse(Int, match(r"step(\d+)", d).captures[1])
for stepdir in sort(filter(d -> isdir(joinpath(vtkdir, d)), readdir(vtkdir)), by=stepnum)
    step = stepnum(stepdir)
    xs, ys, ps = Float64[], Float64[], Float64[]
    for vf in filter(endswith(".vtu"), readdir(joinpath(vtkdir, stepdir), join=true))
        vtk = VTKFile(vf)
        pts = get_points(vtk)
        mask = abs.(pts[3, :] .- z_target) .< 1.0
        any(mask) || continue
        append!(xs, pts[1, mask]); append!(ys, pts[2, mask])
        append!(ps, get_data(get_point_data(vtk)["pressure"])[mask])
    end
    isempty(xs) && continue
    outfile = joinpath(outdir, "p_step$(lpad(step, 7, '0'))")
    vtk_grid(outfile, xs, ys, fill(z_target, length(xs)); compress=true) do vtkfile
        vtkfile["pressure"] = ps
    end
    println("step $step: $(length(xs)) pts")
end
