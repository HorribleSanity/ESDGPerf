function volume_knl(::VolumeKernel{0},
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
  l_q1 = MVector{nstate, FT}(undef)
  l_q2 = MVector{nstate, FT}(undef)
  l_aux1 = MVector{naux, FT}(undef)
  l_aux2 = MVector{naux, FT}(undef)

  s_G = @cuStaticSharedMem(FT, (Nq3, 3))
  s_D = @cuStaticSharedMem(FT, (Nq, Nq))

  ijk = i + Nq * (j - 1 + Nq * (k - 1))

  @inbounds @fastmath begin
    @unroll for s in 1:nstate
      l_dq[s] = dq[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:nstate
      l_q1[s] = q[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:naux
      l_aux1[s] = aux[ijk + (s - 1) * Nq3 + Nq3 * naux * (e - 1)]
    end

    MJ = vgeo[ijk + (_MJ - 1) * Nq3 + Nq3 * nvgeo * (e - 1)]

    @unroll for d in 1:3
      s_G[ijk, d] =  MJ * vgeo[ijk + (_g11 - 1 + dir - 1 + 3 * (d - 1)) * Nq3 + Nq3 * nvgeo * (e - 1)]
    end

    s_D[i, j] = D[i + Nq * (j - 1)]
    MJI = 1 / MJ

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

      @unroll for s in 1:nstate
        l_q2[s] = q[ljk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
      end

      @unroll for s in 1:naux
        l_aux2[s] = aux[ljk + (s - 1) * Nq3 + Nq3 * naux * (e - 1)]
      end

      volume_flux(l_H, l_q1, l_aux1, l_q2, l_aux2)

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

@inline ave(a, b) = @fastmath (a + b) / 2
@inline function logave(a::FT, b::FT) where {FT}
    #@fastmath begin
      zeta = a / b
      f = (zeta - 1) / (zeta + 1)
      u = f * f

      if u < eps(FT)
          u2 = u * u
          u3 = u2 * u
          u4 = u2 * u2
          F = u4 / 9 + u3 / 7 + u2 / 5 + u / 3 + 1;
      else
          F = log(zeta) / (2 * f)
      end

      return (a + b) / (2 * F)
    #end
end

@inline function pressure(ρ, ρu, ρv, ρw, ρe, Φ)
  @fastmath begin
    FT = typeof(ρ)
    (_γ(FT) - 1) * (ρe - (ρu * ρu + ρv * ρv + ρw * ρw) / 2ρ - ρ * Φ)
  end
end

@inline function volume_flux(H, state1, aux1, state2, aux2)
  FT = eltype(H)

  @inbounds @fastmath begin
    ρ1 = state1[_ρ]
    ρu1 = state1[_ρu]
    ρv1 = state1[_ρv]
    ρw1 = state1[_ρw]
    ρe1 = state1[_ρe]
    Φ1 = aux1[_Φ]

    u1 = ρu1 / ρ1
    v1 = ρv1 / ρ1
    w1 = ρw1 / ρ1
    p1 = pressure(ρ1, ρu1, ρv1, ρw1, ρe1, Φ1)
    b1 = ρ1 / (2 * p1)
    usq1 = u1 * u1 + v1 * v1 + w1 * w1

    ρ2 = state2[_ρ]
    ρu2 = state2[_ρu]
    ρv2 = state2[_ρv]
    ρw2 = state2[_ρw]
    ρe2 = state2[_ρe]
    Φ2 = aux2[_Φ]

    u2 = ρu2 / ρ2
    v2 = ρv2 / ρ2
    w2 = ρw2 / ρ2
    p2 = pressure(ρ2, ρu2, ρv2, ρw2, ρe2, Φ2)
    b2 = ρ2 / (2 * p2)
    usq2 = u2 * u2 + v2 * v2 + w2 * w2

    r_avg = ave(ρ1, ρ2)
    u_avg = ave(u1, u2)
    v_avg = ave(v1, v2)
    w_avg = ave(w1, w2)
    b_avg = ave(b1, b2)
    Φ_avg = ave(Φ1, Φ2)
    usq_avg = ave(usq1, usq2)

    r_log = logave(ρ1, ρ2)
    b_log = logave(b1, b2)
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
