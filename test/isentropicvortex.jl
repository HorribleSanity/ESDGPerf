using Raven
using ESDGPerf
using CUDA
using StaticArrays
using Printf
using Logging
using Test

@inline function vortex!(state, aux, x⃗, t)
  FT = eltype(state)
  x⃗₂ = x⃗[StaticArrays.SOneTo(2)]
  γ = _γ(FT)
  c⃗ = SVector(FT(5), FT(0))
  u⃗₀ = SVector(FT(1), FT(0))
  β = FT(5)
  r⃗ = x⃗₂ - c⃗ - u⃗₀ * t
  f = β * exp(1 - r⃗' * r⃗)

  ρ = (1 - (γ - 1) * f ^ 2 / (16 * γ * π ^ 2)) ^ (1 / (γ - 1))
  u⃗ = u⃗₀ + f / 2π * SVector(-r⃗[2], r⃗[1])
  ρu⃗ = ρ * u⃗
  p = ρ ^ γ
  Φ = 0
  ρe = p / (γ - 1) + ρu⃗' * ρu⃗ / 2ρ + ρ * Φ

  state[_ρ] = ρ
  state[_ρu] = ρu⃗[1]
  state[_ρv] = ρu⃗[2]
  state[_ρw] = 0
  state[_ρe] = ρe

  aux[_Φ] = Φ
end

function run(kernel, FT, N, K)
  A = CuArray
  Nq = N + 1

  cell = LobattoCell{Tuple{Nq, Nq, Nq}, FT, A}()

  vx = range(FT(0), stop=FT(20), length=K+1)
  vy = range(FT(-5), stop=FT(5), length=K+1)
  vz = range(FT(-5), stop=FT(5), length=2)

  function meshwarp(x)
      x₁, x₂, x₃ = x
      x̃₁ = x₁ + sinpi(x₁ / 20) * sinpi(2 * (x₂ + 5) / 10) * 4
      x̃₂ = x₂ - sinpi(2 * x₁ / 20) * sinpi((x₂ + 5) / 10) * 2
      x̃₃ = x₃
      SVector(x̃₁, x̃₂, x̃₃)
  end
  cg = brick((vx, vy, vz), (true, true, true))
  gm = GridManager(cell, cg)
  grid = generate(meshwarp, gm)

  vgeo, sgeo = geometric_factors(kernel, grid)
  MJ = view(vgeo, :, _MJ:_MJ, :)

  ncells = length(grid)
  np = length(cell)
  q = CuArray{FT}(undef, np, nstate, ncells)
  aux = CuArray{FT}(undef, np, naux, ncells)

  initialize!(vortex!, q, aux, vgeo, FT(0))

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

  for _ = 1:numberofsteps
    for stage = eachindex(RKA, RKB)
      dq .*= RKA[stage]
      rhs!(kernel, dq, q, aux, vgeo, sgeo, cell, grid)
      q .+= RKB[stage] * dt * dq
    end
  end

  qexact = similar(q)
  initialize!(vortex!, qexact, aux, vgeo, timeend)

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
  nlevels = 3

  expected_error = Dict()
  expected_error[1] = 1.3808132150048602e+01
  expected_error[2] = 6.2418885314721635e+00
  expected_error[3] = 8.4685056013974602e-01

  for kernel in all_kernels
    errs = zeros(FT, nlevels)
    for l in 1:nlevels
      K = 5 * 2 ^ (l - 1)
      errs[l] = run(kernel, FT, N, K)
      @test errs[l] ≈ expected_error[l]
    end
    if nlevels > 1
      rates = log2.(errs[1:(nlevels-1)] ./ errs[2:nlevels])
      @info "Convergence rates\n" *
        join(["rate for levels $l → $(l + 1) = $(rates[l])" for l in 1:(nlevels - 1)], "\n")
    end
  end
end
