using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using Test

@inline function uniform!(q, aux, x⃗, t)
  FT = eltype(q)
  γ = _γ(FT)

  ρ = FT(1000)
  ρu⃗ = ρ .* SVector(5, 0, 10)
  p = FT(1000)
  Φ = FT(0)
  ρe = p / (γ - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ

  q[_ρ] = ρ
  q[_ρu] = ρu⃗[1]
  q[_ρv] = ρu⃗[2]
  q[_ρw] = ρu⃗[3]
  q[_ρe] = ρe

  aux[_Φ] = Φ
end

function run(kernel, FT, N, K)
  A = CuArray
  Nq = N + 1

  cell = LobattoCell{FT, A}(Nq, Nq, Nq)

  vx = range(FT(0), stop=FT(20), length=K+1)
  vy = range(FT(-5), stop=FT(5), length=K+1)
  vz = range(FT(-5), stop=FT(5), length=2)
  grid = brickgrid(cell, (vx, vy, vz); periodic=(true, true, true))

  vgeo, sgeo = geometric_factors(kernel, cell, grid)
  MJ = view(vgeo, :, _MJ:_MJ, :)

  ncells = length(grid)
  np = length(cell)
  q = CuArray{FT}(undef, np, nstate, ncells)
  aux = CuArray{FT}(undef, np, naux, ncells)

  initialize!(uniform!, q, aux, vgeo, FT(0))

  dq = similar(q)
  fill!(dq, FT(0))

  timeend = FT(5)
  cfl = FT(1 // 4)
  dt = cfl * min_node_distance(grid)
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

  en0 = sqrt(sum(@. MJ * q ^ 2))
  @info @sprintf """Starting
  N        = %d
  K        = %d
  kernel   = %s
  norm(q)  = %.16e
  """ N K kernel en0

  for step = 1:numberofsteps
    for stage = 1:length(RKA)
      dq .*= RKA[stage]
      rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid)
      q .+= RKB[stage] * dt * dq
    end
  end

  qexact = similar(q)
  initialize!(uniform!, qexact, aux, vgeo, timeend)

  enf = sqrt(sum(@. MJ * q ^ 2))
  errf = sqrt(sum(@. vgeo[:, _MJ:_MJ, :] * (q - qexact) ^ 2))

  @info @sprintf """Finished
  norm(q)      = %.16e
  norm(q - qe) = %.16e
  """ enf errf

  errf
end

let
  FT = Float64
  N  = 4
  K = 10
  nlevels = 2

  expected_error = 1.7457299184063730e-07

  for kernel in all_kernels
    err = run(kernel, FT, N, K)
    @test err ≈ expected_error atol=10sqrt(eps(FT))
  end
end
