const _MJI = 15
const _logρ = 6
const _logb = 7

function kernel_vgeo(::VolumeKernel{:final}, vgeo)
  MJ = view(vgeo, :, _MJ:_MJ, :)
  MJI = 1 ./ MJ
  G = view(vgeo, :, _g11:_g11+9, :)
  G .*= MJ
  cat(vgeo, MJI, dims=2)
end

function volume_knl(::VolumeKernel{:final},
                    dq, q, aux, vgeo, D,
                    ::Val{Nq}, ::Val{nvgeo}, ::Val{dir}) where {Nq, nvgeo, dir}
  FT = eltype(dq)
  Nq3 = Nq ^ 3

  e = blockIdx().x
  i = threadIdx().x
  j = threadIdx().y
  k = threadIdx().z

  l_H = MMatrix{3, nstate, FT}(undef)
  l_dq = MVector{nstate, FT}(undef)
  l_q1 = MVector{nstate+2, FT}(undef)
  l_q2 = MVector{nstate+2, FT}(undef)
  l_aux1 = MVector{naux, FT}(undef)
  l_aux2 = MVector{naux, FT}(undef)

  s_G = CuStaticSharedArray(FT, (Nq3, 3))
  s_q = CuStaticSharedArray(FT, (Nq3, nstate+2))
  s_D = CuStaticSharedArray(FT, (Nq, Nq))

  ijk = i + Nq * (j - 1 + Nq * (k - 1))

  @inbounds begin
    @unroll for s in 1:nstate
      l_dq[s] = dq[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:nstate
      l_q1[s] = q[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:naux
      l_aux1[s] = aux[ijk + (s - 1) * Nq3 + Nq3 * naux * (e - 1)]
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

    MJI = vgeo[ijk + (_MJI - 1) * Nq3 + Nq3 * nvgeo * (e - 1)]

    @unroll for d in 1:3
      s_G[ijk, d] = vgeo[ijk + (_g11 - 1 + dir - 1 + 3 * (d - 1)) * Nq3 + Nq3 * nvgeo * (e - 1)]
    end

    s_D[i, j] = D[i + Nq * (j - 1)]

    sync_threads()

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

      @unroll for s in 1:nstate+2
        l_q2[s] = s_q[ljk, s]
      end

      @unroll for s in 1:naux
        l_aux2[s] = aux[ljk + (s - 1) * Nq3 + Nq3 * naux * (e - 1)]
      end

      volume_flux_prim(l_H, l_q1, l_aux1, l_q2, l_aux2)

      @unroll for s in 1:nstate
        G1_ljk = s_G[ljk, 1]
        G2_ljk = s_G[ljk, 2]
        G3_ljk = s_G[ljk, 3]

        G1_ijk = s_G[ijk, 1]
        G2_ijk = s_G[ijk, 2]
        G3_ijk = s_G[ijk, 3]

        l_dq[s] -=
          MJI *
          s_D[id, l] *
          (
              G1_ijk * l_H[1, s] +
              G2_ijk * l_H[2, s] +
              G3_ijk * l_H[3, s]
          )

        l_dq[s] +=
          MJI *
          (
              l_H[1, s] * G1_ljk +
              l_H[2, s] * G2_ljk +
              l_H[3, s] * G3_ljk
          ) *
          s_D[l, id]
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
  begin
    FT = typeof(ρ)
    (_γ(FT) - 1) * ρ * (e - (u * u + v * v + w * w) / 2 - Φ)
  end
end

@inline function volume_flux_prim(H, q1, aux1, q2, aux2)
  FT = eltype(H)

  @inbounds begin
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

    r_avg = ave(ρ1, ρ2)
    u_avg = ave(u1, u2)
    v_avg = ave(v1, v2)
    w_avg = ave(w1, w2)
    b_avg = ave(b1, b2)
    Φ_avg = ave(Φ1, Φ2)
    usq_avg = ave(usq1, usq2)

    r_log = logave_fast(ρ1, ρ2, logρ1, logρ2)
    b_log = logave_fast(b1, b2, logb1, logb2)
    alpha = b_avg * r_log / (2 * b1)

    H[1, _ρ] = u_avg * r_log
    H[2, _ρ] = v_avg * r_log
    H[3, _ρ] = w_avg * r_log

    p = r_avg / (2 * b_avg)
    D = -alpha * (Φ1 - Φ2)

    H[1, _ρu] = u_avg * H[1, _ρ] + p
    H[2, _ρu] = v_avg * H[1, _ρ]
    H[3, _ρu] = w_avg * H[1, _ρ]

    H[1, _ρv] = u_avg * H[2, _ρ]
    H[2, _ρv] = v_avg * H[2, _ρ] + p
    H[3, _ρv] = w_avg * H[2, _ρ]

    H[1, _ρw] = u_avg * H[3, _ρ]
    H[2, _ρw] = v_avg * H[3, _ρ]
    H[3, _ρw] = w_avg * H[3, _ρ] + p

    c = (1 / (2 * (_γ(FT) - 1) * b_log) - usq_avg / 2 + Φ_avg)

    c1 = H[1, _ρu] * u_avg + H[1, _ρv] * v_avg + H[1, _ρw] * w_avg
    c2 = H[2, _ρu] * u_avg + H[2, _ρv] * v_avg + H[2, _ρw] * w_avg
    c3 = H[3, _ρu] * u_avg + H[3, _ρv] * v_avg + H[3, _ρw] * w_avg
    H[1, _ρe] = c * H[1, _ρ] + c1
    H[2, _ρe] = c * H[2, _ρ] + c2
    H[3, _ρe] = c * H[3, _ρ] + c3

    H[1, _ρu] += D
    H[2, _ρv] += D
    H[3, _ρw] += D
  end
end
