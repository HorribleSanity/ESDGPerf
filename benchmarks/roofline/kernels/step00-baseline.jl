const _MJI = 15

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
  l_q1 = MVector{nstate, FT}(undef)
  l_q2 = MVector{nstate, FT}(undef)
  l_aux1 = MVector{naux, FT}(undef)
  l_aux2 = MVector{naux, FT}(undef)

  s_G = CuStaticSharedArray(FT, (Nq3, 3))
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
