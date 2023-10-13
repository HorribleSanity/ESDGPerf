using Raven
using ESDGPerf
using CUDA
using StaticArrays

@inline function init!(q, aux, x⃗, t)
  FT = eltype(q)
  γ = _γ(FT)
  r = sqrt(x⃗' * x⃗)

  δρ = r <= 1 ? FT(1 / 10) * cos(π * r / 2) : 0

  ρ = FT(1) + δρ
  u⃗ = SVector{3, FT}(1, 2, 3)
  ρu⃗ = ρ * u⃗
  p = FT(1)
  Φ = FT(1)
  ρe = p / (γ - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ

  q[_ρ] = ρ
  q[_ρu] = ρu⃗[1]
  q[_ρv] = ρu⃗[2]
  q[_ρw] = ρu⃗[3]
  q[_ρe] = ρe

  aux[_Φ] = Φ
end

function profile(kernel, FT, N, K)
  A = CuArray
  Nq = N + 1

  cell = LobattoCell{Tuple{Nq, Nq, Nq}, FT, A}()

  vx = range(FT(-5), stop=FT(5), length=K+1)
  vy = range(FT(-5), stop=FT(5), length=K+1)
  vz = range(FT(-5), stop=FT(5), length=K+1)

  cg = brick((vx, vy, vz), (true, true, true))
  gm = GridManager(cell, cg)
  grid = generate(gm)

  vgeo, _ = geometric_factors(kernel, grid)

  ncells = length(grid)
  np = length(cell)
  q = CuArray{FT}(undef, np, nstate, ncells)
  aux = CuArray{FT}(undef, np, naux, ncells)

  initialize!(init!, q, aux, vgeo, FT(0))

  dq = similar(q)
  fill!(dq, FT(0))

  @CUDA.profile launch_volume(kernel, dq, q, aux, vgeo, cell, grid)
  elap = @CUDA.elapsed launch_volume(kernel, dq, q, aux, vgeo, cell, grid)
  println("Elapsed time: $elap s")
end

let
  FT = Float64
  N  = 4
  K = 60
  for kernel in all_kernels
    profile(kernel, FT, N, K)
  end
end
