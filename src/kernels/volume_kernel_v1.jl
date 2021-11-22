const _MJI = 15
const _logρ = 6
const _logb = 7

directionally_split(::VolumeKernel{1}) = false

function kernel_vgeo(::VolumeKernel{1}, vgeo)
  MJ = view(vgeo, :, _MJ:_MJ, :)
  MJI = 1 ./ MJ
  G = view(vgeo, :, _g11:_g11+9, :)
  G .*= MJ
  cat(vgeo, MJI, dims=2)
end

function volume_knl(::VolumeKernel{1},
                    dq, q, aux, vgeo, D,
                    ::Val{Nq}, ::Val{nvgeo}) where {Nq, nvgeo}
  FT = eltype(dq)
  Nq3 = Nq ^ 3

  e = blockIdx().x
  i = threadIdx().x
  j = threadIdx().y
  k = threadIdx().z

  l_G1 = MVector{3, FT}(undef)
  l_G2 = MVector{3, FT}(undef)
  l_dq = MVector{nstate, FT}(undef)
  l_q1 = MVector{nstate+2, FT}(undef)
  l_q2 = MVector{nstate+2, FT}(undef)
  l_aux1 = MVector{naux, FT}(undef)
  l_aux2 = MVector{naux, FT}(undef)

  s_q = @cuStaticSharedMem(FT, (Nq3, nstate+2))
  s_aux = @cuStaticSharedMem(FT, (Nq3, naux))
  s_D = @cuStaticSharedMem(FT, (Nq, Nq))

  ijk = i + Nq * (j - 1 + Nq * (k - 1))

  @inbounds begin
    MJI = vgeo[ijk + (_MJI - 1) * Nq3 + Nq3 * nvgeo * (e - 1)]
    s_D[i, j] = D[i + Nq * (j - 1)]

    @unroll for s in 1:nstate
      l_dq[s] = dq[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:nstate
      l_q1[s] = q[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:naux
      l_aux1[s] = aux[ijk + (s - 1) * Nq3 + Nq3 * naux * (e - 1)]
      s_aux[ijk, s] = l_aux1[s]
    end

    iρ = 1 / l_q1[_ρ]
    @unroll for s in 2:nstate
      l_q1[s] *= iρ
    end
    p = pressure_prim(l_q1[_ρ], l_q1[_ρu], l_q1[_ρv], l_q1[_ρw], l_q1[_ρe], l_aux1[_Φ])
    ip = FT(1) / p
    b = FT(1 / 2) * l_q1[_ρ] * ip

    l_q1[_ρe] = b
    l_q1[_logρ] = log(l_q1[_ρ])
    l_q1[_logb] = log(b)

    @unroll for s in 1:nstate+2
      s_q[ijk, s] = l_q1[s]
    end


    sync_threads()

    @unroll 1 for dir in 1:3
      @unroll 1 for l in 1:Nq
        if dir == 1
          id = i
          ljk = l + Nq * ((j - 1) + Nq * (k - 1))
        elseif dir == 2
          id = j
          ljk = i + Nq * ((l - 1) + Nq * (k - 1))
        elseif dir == 3
          id = k
          ljk = i + Nq * ((j - 1) + Nq * (l - 1))
        end

        Didl = s_D[id, l]
        Dlid = s_D[l, id]

        @unroll for s in 1:nstate+2
          l_q2[s] = s_q[ljk, s]
        end

        @unroll for s in 1:naux
          l_aux2[s] = s_aux[ljk, s]
        end

        @unroll for d in 1:3
          l_G1[d] = vgeo[ijk + (_g11 - 1 + dir - 1 + 3 * (d - 1)) * Nq3 + Nq3 * nvgeo * (e - 1)]
          l_G2[d] = vgeo[ljk + (_g11 - 1 + dir - 1 + 3 * (d - 1)) * Nq3 + Nq3 * nvgeo * (e - 1)]
        end

        volume_dq_prim(l_dq, Didl, Dlid, MJI, l_G1, l_G2, l_q1, l_aux1, l_q2, l_aux2)
      end
    end

    @unroll for s in 1:nstate
      dq[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)] = l_dq[s]
    end
  end
  nothing
end


@inline function logave_fast(a::FT, b::FT, loga::FT, logb::FT) where {FT}
    amb = a - b
    apb = a + b
    if abs(amb) < sqrt(eps(FT)) * abs(apb)
      c1 = FT(2) / FT(3)
      c2 = FT(2) / FT(5)
      c3 = FT(2) / FT(7)
      c4 = FT(2) / FT(9)

      iapb = FT(1) / apb
      u = amb ^ 2 * iapb ^ 2

      F = 2 + u * (c1 + u * (c2 + u * (c3 + u * c4)))
      iF = FT(1) / F
      return apb * iF
    else
      ilog = FT(1) / (loga - logb)
      return amb * ilog
    end
end


@inline function pressure_prim(ρ, u, v, w, e, Φ)
  @fastmath begin
    FT = typeof(ρ)
    (_γ(FT) - 1) * ρ * (e - (u * u + v * v + w * w) / 2 - Φ)
  end
end

@inline function volume_dq_prim(l_dq, Didl, Dlid, MJI, G1, G2, q1, aux1, q2, aux2)
  FT = eltype(l_dq)

  @inbounds @fastmath begin
    ρ1 =  q1[_ρ]
    u1 = q1[_ρu]
    v1 = q1[_ρv]
    w1 = q1[_ρw]
    b1 = q1[_ρe]
    logρ1 = q1[_logρ]
    logb1 = q1[_logb]
    Φ1 = aux1[_Φ]

    usq1 = u1 * u1 + v1 * v1 + w1 * w1

    ρ2 =  q2[_ρ]
    u2 = q2[_ρu]
    v2 = q2[_ρv]
    w2 = q2[_ρw]
    b2 = q2[_ρe]
    logρ2 = q2[_logρ]
    logb2 = q2[_logb]
    Φ2 = aux2[_Φ]

    usq2 = u2 * u2 + v2 * v2 + w2 * w2

    ρ_avg = ave(ρ1, ρ2)
    u_avg = ave(u1, u2)
    v_avg = ave(v1, v2)
    w_avg = ave(w1, w2)
    b_avg = ave(b1, b2)
    Φ_avg = ave(Φ1, Φ2)
    usq_avg = ave(usq1, usq2)

    ρ_log = logave_fast(ρ1, ρ2, logρ1, logρ2)
    b_log = logave_fast(b1, b2, logb1, logb2)

    alpha = b_avg * ρ_log / (2 * b1)
    p = ρ_avg / (2 * b_avg)
    D = -alpha * (Φ1 - Φ2)
    c = (1 / (2 * (_γ(FT) - 1) * b_log) - usq_avg / 2 + Φ_avg)

    ρuhat1 = G1[1] * u_avg + G1[2] * v_avg + G1[3] * w_avg
    ρuhat1 *= ρ_log

    H1ijk_ρ = ρuhat1
    H1ijk_ρu = ρuhat1 * u_avg + p * G1[1]
    H1ijk_ρv = ρuhat1 * v_avg + p * G1[2]
    H1ijk_ρw = ρuhat1 * w_avg + p * G1[3]
    H1ijk_ρe = c * H1ijk_ρ + H1ijk_ρu * u_avg + H1ijk_ρv * v_avg + H1ijk_ρw * w_avg

    H1ijk_ρu += G1[1] * D
    H1ijk_ρv += G1[2] * D
    H1ijk_ρw += G1[3] * D

    H1ijk_ρ *= Didl
    H1ijk_ρu *= Didl
    H1ijk_ρv *= Didl
    H1ijk_ρw *= Didl
    H1ijk_ρe *= Didl

    ρuhat2 = G2[1] * u_avg + G2[2] * v_avg + G2[3] * w_avg
    ρuhat2 *= ρ_log

    H2ijk_ρ = ρuhat2
    H2ijk_ρu = ρuhat2 * u_avg + p * G2[1]
    H2ijk_ρv = ρuhat2 * v_avg + p * G2[2]
    H2ijk_ρw = ρuhat2 * w_avg + p * G2[3]
    H2ijk_ρe = c * H2ijk_ρ + H2ijk_ρu * u_avg + H2ijk_ρv * v_avg + H2ijk_ρw * w_avg

    H2ijk_ρu += G2[1] * D
    H2ijk_ρv += G2[2] * D
    H2ijk_ρw += G2[3] * D

    H2ijk_ρ  *= Dlid
    H2ijk_ρu *= Dlid
    H2ijk_ρv *= Dlid
    H2ijk_ρw *= Dlid
    H2ijk_ρe *= Dlid

    l_dq[_ρ] += MJI * (H2ijk_ρ  - H1ijk_ρ)
    l_dq[_ρu] += MJI * (H2ijk_ρu  - H1ijk_ρu)
    l_dq[_ρv] += MJI * (H2ijk_ρv  - H1ijk_ρv)
    l_dq[_ρw] += MJI * (H2ijk_ρw  - H1ijk_ρw)
    l_dq[_ρe] += MJI * (H2ijk_ρe  - H1ijk_ρe)
  end
end
