function init_knl(initfun!, q, aux, vgeo, t, ::Val{Nq}, ::Val{nvgeo}) where {Nq, nvgeo}
  Nq3 = Nq ^ 3
  FT = eltype(q)

  ijk = threadIdx().x
  e = blockIdx().x

  l_q = MVector{nstate, FT}(undef)
  l_aux = MVector{naux, FT}(undef)
  x⃗ = MVector{3, FT}(undef)

  @inbounds begin
    @unroll for s in 1:nstate
      l_q[s] = q[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)]
    end

    @unroll for s in 1:naux
      l_aux[s] = aux[ijk + Nq3 * (s - 1) + Nq3 * naux * (e - 1)]
    end

    @unroll for d in 1:3
      x⃗[d] = vgeo[ijk + Nq3 * (_x + d - 2) + Nq3 * nvgeo * (e - 1)]
    end

    initfun!(l_q, l_aux, SVector(x⃗), t)

    @unroll for s in 1:nstate
      q[ijk + Nq3 * (s - 1) + Nq3 * nstate * (e - 1)] = l_q[s]
    end

    @unroll for s in 1:naux
      aux[ijk + Nq3 * (s - 1) + Nq3 * naux * (e - 1)] = l_aux[s]
    end
  end

  nothing
end
