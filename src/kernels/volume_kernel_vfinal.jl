const _MJI = 15
const _logρ = 6
const _logb = 7

directionally_split(::VolumeKernel{:final}) = false

function kernel_vgeo(::VolumeKernel{:final}, vgeo)
    MJ = view(vgeo, :, _MJ:_MJ, :)
    MJI = 1 ./ MJ
    G = view(vgeo, :, _g11:_g11+8, :)
    G .*= MJ
    @assert _MJI == size(vgeo, 2) + 1
    cat(vgeo, MJI, dims = 2)
end

function volume_knl(
    ::VolumeKernel{:final},
    dq,
    q,
    aux,
    vgeo,
    D,
    ::Val{Nq},
    ::Val{nvgeo},
) where {Nq,nvgeo}
    FT = eltype(dq)
    Nq3 = Nq^3

    vgeo = Base.Experimental.Const(vgeo)

    e = blockIdx().x
    i = threadIdx().x
    j = threadIdx().y
    k = threadIdx().z

    l_dq_ijk = MVector{nstate,FT}(undef)
    l_q1 = MVector{nstate + 2,FT}(undef)
    #l_q2 = MVector{nstate+2, FT}(undef)
    #l_aux1 = MVector{naux, FT}(undef)
    #l_aux2 = MVector{naux, FT}(undef)

    s_q = CuStaticSharedArray(FT, (Nq3, nstate + 2))
    s_aux = CuStaticSharedArray(FT, (Nq3, naux))
    s_D = CuStaticSharedArray(FT, (Nq, Nq))
    s_dq = CuStaticSharedArray(FT, (Nq3, nstate))

    ijk = i + Nq * (j - 1 + Nq * (k - 1))

    ijk_vgeo = ijk - Nq3 + Nq3 * nvgeo * (e - 1)
    ijk_state = ijk - Nq3 + Nq3 * nstate * (e - 1)
    ijk_aux = ijk - Nq3 + Nq3 * naux * (e - 1)

    @inbounds begin
        MJI = vgeo[ijk_vgeo+_MJI*Nq3]
        s_D[i, j] = D[i+Nq*(j-1)]

        @unroll for s = 1:nstate
            l_dq_ijk[s] = dq[ijk_state+Nq3*s]
        end

        @unroll for s = 1:nstate
            l_q1[s] = q[ijk_state+Nq3*s]
        end

        @unroll for s = 1:naux
            #l_aux1[s] = aux[ijk_aux + s * Nq3]
            s_aux[ijk, s] = aux[ijk_aux+s*Nq3]
        end

        iρ = FT(1) / l_q1[_ρ]
        @unroll for s = 2:nstate
            l_q1[s] *= iρ
        end
        p = pressure_prim(
            l_q1[_ρ],
            l_q1[_ρu],
            l_q1[_ρv],
            l_q1[_ρw],
            l_q1[_ρe],
            s_aux[ijk, _Φ],
        )
        ip = FT(1) / p
        b = FT(1 / 2) * l_q1[_ρ] * ip

        l_q1[_ρe] = b
        l_q1[_logρ] = log(l_q1[_ρ])
        l_q1[_logb] = log(b)

        @unroll for s = 1:nstate+2
            s_q[ijk, s] = l_q1[s]
        end
        @unroll for s = 1:nstate
            s_dq[ijk, s] = -zero(FT)
        end

        sync_threads()

        @unroll for dir = 1:3
            if dir == 1
                id = i
            elseif dir == 2
                id = j
            elseif dir == 3
                id = k
            end

            G_dir = (_g11 + dir - 1 - 3) * Nq3
            ijk_G_dir = ijk_vgeo + G_dir

            @unroll 1 for ll = 2:(Nq÷2+1)
                l = mod(id + ll - 2, Nq) + 1

                if dir == 1
                    ljk = l + Nq * ((j - 1) + Nq * (k - 1))
                elseif dir == 2
                    ljk = i + Nq * ((l - 1) + Nq * (k - 1))
                elseif dir == 3
                    ljk = i + Nq * ((j - 1) + Nq * (l - 1))
                end

                ljk_vgeo = ljk - Nq3 + Nq3 * nvgeo * (e - 1)
                ljk_G_dir = ljk_vgeo + G_dir

                volume_dq_prim(
                    l_dq_ijk,
                    s_dq,
                    s_D,
                    MJI,
                    vgeo,
                    l_q1,
                    s_q,
                    s_aux,
                    ijk,
                    ljk,
                    id,
                    l,
                    ijk_G_dir,
                    ljk_G_dir,
                    3 * Nq3,
                )
            end
        end

        @unroll for s = 1:nstate
            l_dq_ijk[s] += MJI * s_dq[ijk, s]
            dq[ijk_state+Nq3*s] = l_dq_ijk[s]
        end
    end
    nothing
end

ave_fast(a::FT, b::FT) where {FT} = muladd(b, FT(1) / FT(2), a)
@inline function inv_logave_fast(a::FT, b::FT, loga::FT, logb::FT) where {FT}
    a *= 2
    amb = a - b
    apb = a + b
    if abs(amb) < sqrt(logave_ϵ(FT)) * abs(apb)
        c1 = FT(2) / FT(3)
        c2 = FT(2) / FT(5)
        c3 = FT(2) / FT(7)
        c4 = FT(2) / FT(9)

        iapb = FT(1) / apb
        u = amb^2 * iapb^2

        F = 2 + u * (c1 + u * (c2 + u * (c3 + u * c4)))
        return iapb * F
    else
        iamb = FT(1) / amb
        return iamb * (loga - logb)
    end
end
@inline function inv_logave_fast(a::FT, b::FT, loga::FT, logb::FT) where {FT}
#     a *= 2
#     amb = a - b
#     apb = a + b
#     if abs(amb) < sqrt(logave_ϵ(FT)) * apb # Get rid of abs
#         c1 = FT(2) / FT(3)
#         c2 = FT(2) / FT(5)
#         c3 = FT(2) / FT(7)
#         c4 = FT(2) / FT(9)
# 
#         iapb = FT(1) / apb
#         u = amb^2 * iapb^2
# 
#         F = 2 + u * (c1 + u * (c2 + u * c3)) # Get rid of c4
#         return iapb * F
#     else
#         iamb = FT(1) / amb
#         return iamb * (loga - logb)
#     end
# end


@inline function logave_fast(a::FT, b::FT, loga::FT, logb::FT) where {FT}
    a *= 2
    amb = a - b
    apb = a + b
    if abs(amb) < sqrt(logave_ϵ(FT)) * abs(apb)
        c1 = FT(2) / FT(3)
        c2 = FT(2) / FT(5)
        c3 = FT(2) / FT(7)
        c4 = FT(2) / FT(9)

        iapb = FT(1) / apb
        u = amb^2 * iapb^2

        F = 2 + u * (c1 + u * (c2 + u * (c3 + u * c4)))
        iF = FT(1) / F
        return apb * iF
    else
        ilog = FT(1) / (loga - logb)
        return amb * ilog
    end
end

# Modified
# @inline function logave_fast(a::FT, b::FT, loga::FT, logb::FT) where {FT}
#     a *= 2 # Possibly not necessary
#     amb = a - b
#     apb = a + b
#     if abs(amb) < sqrt(logave_ϵ(FT)) * apb # Remove absolute value
#         c1 = FT(2) / FT(3)
#         c2 = FT(2) / FT(5)
#         c3 = FT(2) / FT(7)
#         c4 = FT(2) / FT(9)
# 
#         iapb = FT(1) / apb
#         u = amb^2 * iapb^2
# 
#         F = 2 + u * (c1 + u * (c2 + u * c3)) # Taking out c4
#         iF = FT(1) / F
#         return apb * iF
#     else
#         ilog = FT(1) / (loga - logb)
#         return amb * ilog
#     end
# end


@inline function pressure_prim(ρ, u, v, w, e, Φ)
    FT = typeof(ρ)
    (_γ(FT) - 1) * ρ * (e - (u * u + v * v + w * w) / 2 - Φ)
end

@inline function volume_dq_prim(
    dq_ijk,
    s_dq,
    s_D,
    MJI,
    vgeo,
    q1,
    s_q,
    s_aux,
    ijk,
    ljk,
    id,
    l,
    G1base,
    G2base,
    Gstride,
)
    FT = eltype(dq_ijk)

    @inbounds begin
        half = FT(1) / FT(2)

        G11 = vgeo[G1base+Gstride]
        G12 = vgeo[G1base+2Gstride]
        G13 = vgeo[G1base+3Gstride]

        G21 = vgeo[G2base+Gstride]
        G22 = vgeo[G2base+2Gstride]
        G23 = vgeo[G2base+3Gstride]

        ρ1 = half * q1[_ρ]
        u1 = half * q1[_ρu]
        v1 = half * q1[_ρv]
        w1 = half * q1[_ρw]
        b1 = half * q1[_ρe]
        ib1 = FT(1) / b1
        logρ1 = q1[_logρ]
        logb1 = q1[_logb]
        Φ1 = half * s_aux[ijk, _Φ]

        usq1 = 2 * (u1 * u1 + v1 * v1 + w1 * w1)

        ρ2 = s_q[ljk, _ρ]
        u2 = s_q[ljk, _ρu]
        v2 = s_q[ljk, _ρv]
        w2 = s_q[ljk, _ρw]
        b2 = s_q[ljk, _ρe]
        ib2 = FT(1) / b2
        logρ2 = s_q[ljk, _logρ]
        logb2 = s_q[ljk, _logb]
        Φ2 = s_aux[ljk, _Φ]

        usq2 = u2 * u2 + v2 * v2 + w2 * w2

        ρ_avg = ave_fast(ρ1, ρ2)
        u_avg = ave_fast(u1, u2)
        v_avg = ave_fast(v1, v2)
        w_avg = ave_fast(w1, w2)
        b_avg = ave_fast(b1, b2)
        ib_avg = FT(1) / b_avg
        fac = 2 * b1 * ib2
        Φ_avg = ave_fast(Φ1, Φ2)
        usq_avg = ave_fast(usq1, usq2)

        ρ_log = logave_fast(ρ1, ρ2, logρ1, logρ2)
        #ib_log = inv_logave_fast(b1, b2, logb1, logb2)
        ib_log = inv_logave_fast(b1, b2, logb1, logb2)
        #ρ_log = logave(2*ρ1, ρ2)
        #ib_log = FT(1) / logave(2*b1, b2)

        p = ρ_avg * ib_avg / 2
        igm1h = FT(1) / (2 * (_γ(FT) - 1))
        c = (ib_log * igm1h - usq_avg / 2 + Φ_avg)
        alpha = b_avg * ρ_log * ib1 / 4
        D = -alpha * (2 * Φ1 - Φ2)
        Dfac = -D * fac

        ρuhat1 = G11 * u_avg + G12 * v_avg + G13 * w_avg
        ρuhat1 *= ρ_log

        H1ijk_ρ = ρuhat1
        H1ijk_ρu = ρuhat1 * u_avg + p * G11
        H1ijk_ρv = ρuhat1 * v_avg + p * G12
        H1ijk_ρw = ρuhat1 * w_avg + p * G13
        H1ijk_ρe = c * H1ijk_ρ + H1ijk_ρu * u_avg + H1ijk_ρv * v_avg + H1ijk_ρw * w_avg

        H1ljk_ρu = H1ijk_ρu + Dfac * G11
        H1ljk_ρv = H1ijk_ρv + Dfac * G12
        H1ljk_ρw = H1ijk_ρw + Dfac * G13

        H1ijk_ρu += G11 * D
        H1ijk_ρv += G12 * D
        H1ijk_ρw += G13 * D

        Didl = s_D[id, l]
        Dlid = s_D[l, id]

        H1ijk_ρ *= Didl
        H1ijk_ρu *= Didl
        H1ijk_ρv *= Didl
        H1ijk_ρw *= Didl
        H1ijk_ρe *= Didl

        H1ljk_ρu *= Didl
        H1ljk_ρv *= Didl
        H1ljk_ρw *= Didl

        ρuhat2 = G21 * u_avg + G22 * v_avg + G23 * w_avg
        ρuhat2 *= ρ_log

        H2ijk_ρ = ρuhat2
        H2ijk_ρu = ρuhat2 * u_avg + p * G21
        H2ijk_ρv = ρuhat2 * v_avg + p * G22
        H2ijk_ρw = ρuhat2 * w_avg + p * G23
        H2ijk_ρe = c * H2ijk_ρ + H2ijk_ρu * u_avg + H2ijk_ρv * v_avg + H2ijk_ρw * w_avg

        H2ljk_ρu = H2ijk_ρu + Dfac * G21
        H2ljk_ρv = H2ijk_ρv + Dfac * G22
        H2ljk_ρw = H2ijk_ρw + Dfac * G23

        H2ijk_ρu += G21 * D
        H2ijk_ρv += G22 * D
        H2ijk_ρw += G23 * D

        H2ijk_ρ *= Dlid
        H2ijk_ρu *= Dlid
        H2ijk_ρv *= Dlid
        H2ijk_ρw *= Dlid
        H2ijk_ρe *= Dlid

        H2ljk_ρu *= Dlid
        H2ljk_ρv *= Dlid
        H2ljk_ρw *= Dlid

        H21ijk_ρ = H2ijk_ρ - H1ijk_ρ
        H21ijk_ρe = H2ijk_ρe - H1ijk_ρe

        dq_ijk[_ρ] += MJI * H21ijk_ρ
        dq_ijk[_ρu] += MJI * (H2ijk_ρu - H1ijk_ρu)
        dq_ijk[_ρv] += MJI * (H2ijk_ρv - H1ijk_ρv)
        dq_ijk[_ρw] += MJI * (H2ijk_ρw - H1ijk_ρw)
        dq_ijk[_ρe] += MJI * H21ijk_ρe

        sync_threads()
        s_dq[ljk, _ρ] -= H21ijk_ρ
        s_dq[ljk, _ρu] += (-H2ljk_ρu + H1ljk_ρu)
        s_dq[ljk, _ρv] += (-H2ljk_ρv + H1ljk_ρv)
        s_dq[ljk, _ρw] += (-H2ljk_ρw + H1ljk_ρw)
        s_dq[ljk, _ρe] -= H21ijk_ρe
        sync_threads()
    end
end
