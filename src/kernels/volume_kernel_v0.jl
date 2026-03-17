function volume_knl(
    ::VolumeKernel{0},
    dq,
    q,
    aux,
    vgeo,
    D,
    ::Val{Nq},
    ::Val{nvgeo},
    ::Val{dir},
) where {Nq,nvgeo,dir}
    FT = eltype(dq)
    Nq3 = Nq^3

    e = blockIdx().x
    i = threadIdx().x
    j = threadIdx().y
    k = threadIdx().z

    l_H = MMatrix{3,nstate,FT}(undef)
    l_dq = MVector{nstate,FT}(undef)
    l_q1 = MVector{nstate,FT}(undef)
    l_q2 = MVector{nstate,FT}(undef)
    l_aux1 = MVector{naux,FT}(undef)
    l_aux2 = MVector{naux,FT}(undef)

    s_G = CuStaticSharedArray(FT, (Nq3, 3))
    s_D = CuStaticSharedArray(FT, (Nq, Nq))

    ijk = i + Nq * (j - 1 + Nq * (k - 1))

    @inbounds begin
        @unroll for s = 1:nstate
            l_dq[s] = dq[ijk+Nq3*(s-1)+Nq3*nstate*(e-1)]
        end

        @unroll for s = 1:nstate
            l_q1[s] = q[ijk+Nq3*(s-1)+Nq3*nstate*(e-1)]
        end

        @unroll for s = 1:naux
            l_aux1[s] = aux[ijk+(s-1)*Nq3+Nq3*naux*(e-1)]
        end

        MJ = vgeo[ijk+(_MJ-1)*Nq3+Nq3*nvgeo*(e-1)]

        @unroll for d = 1:3
            s_G[ijk, d] = MJ * vgeo[ijk+(_g11-1+dir-1+3*(d-1))*Nq3+Nq3*nvgeo*(e-1)]
        end

        s_D[i, j] = D[i+Nq*(j-1)]
        MJI = FT(1) / MJ

        sync_threads()

        @unroll 1 for l = 1:Nq
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

            @unroll for s = 1:nstate
                l_q2[s] = q[ljk+Nq3*(s-1)+Nq3*nstate*(e-1)]
            end

            @unroll for s = 1:naux
                l_aux2[s] = aux[ljk+(s-1)*Nq3+Nq3*naux*(e-1)]
            end

            volume_flux(l_H, l_q1, l_aux1, l_q2, l_aux2)

            @unroll for s = 1:nstate
                G1_ljk = s_G[ljk, 1]
                G2_ljk = s_G[ljk, 2]
                G3_ljk = s_G[ljk, 3]

                G1_ijk = s_G[ijk, 1]
                G2_ijk = s_G[ijk, 2]
                G3_ijk = s_G[ijk, 3]

                l_dq[s] -=
                    MJI *
                    s_D[id, l] *
                    (G1_ijk * l_H[1, s] + G2_ijk * l_H[2, s] + G3_ijk * l_H[3, s])

                l_dq[s] +=
                    MJI *
                    (l_H[1, s] * G1_ljk + l_H[2, s] * G2_ljk + l_H[3, s] * G3_ljk) *
                    s_D[l, id]
            end
        end

        @unroll for s = 1:nstate
            dq[ijk+Nq3*(s-1)+Nq3*nstate*(e-1)] = l_dq[s]
        end

    end
    nothing
end

ave(a::FT, b::FT) where {FT} = (a + b) / FT(2)
@inline function logave(a::FT, b::FT) where {FT}
    zeta = a / b
    f = (zeta - FT(1)) / (zeta + FT(1))
    u = f * f

    if u < logave_ϵ(FT)
        u2 = u * u
        u3 = u2 * u
        u4 = u2 * u2
        F = u4 / FT(9) + u3 / FT(7) + u2 / FT(5) + u / FT(3) + FT(1)
    else
        F = log(zeta) / (FT(2) * f)
    end

    return (a + b) / (FT(2) * F)
end

@inline function pressure(ρ, ρu, ρv, ρw, ρe, Φ)
    FT = typeof(ρ)
    (_γ(FT) - FT(1)) * (ρe - (ρu * ρu + ρv * ρv + ρw * ρw) / (FT(2) * ρ) - ρ * Φ)
end

@inline function volume_flux(H, q1, aux1, q2, aux2, n⃗)
    FT = eltype(H)

    @inbounds begin
        ρ1 = q1[_ρ]
        ρu1 = q1[_ρu]
        ρv1 = q1[_ρv]
        ρw1 = q1[_ρw]
        ρe1 = q1[_ρe]
        Φ1 = aux1[_Φ]

        u1 = ρu1 / ρ1
        v1 = ρv1 / ρ1
        w1 = ρw1 / ρ1
        p1 = pressure(ρ1, ρu1, ρv1, ρw1, ρe1, Φ1)
        b1 = ρ1 / (FT(2) * p1)
        usq1 = u1 * u1 + v1 * v1 + w1 * w1

        ρ2 = q2[_ρ]
        ρu2 = q2[_ρu]
        ρv2 = q2[_ρv]
        ρw2 = q2[_ρw]
        ρe2 = q2[_ρe]
        Φ2 = aux2[_Φ]

        u2 = ρu2 / ρ2
        v2 = ρv2 / ρ2
        w2 = ρw2 / ρ2
        p2 = pressure(ρ2, ρu2, ρv2, ρw2, ρe2, Φ2)
        b2 = ρ2 / (FT(2) * p2)
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
        alpha = b_avg * r_log / (FT(2) * b1)

        H[1, _ρ] = u_avg * r_log
        H[2, _ρ] = v_avg * r_log
        H[3, _ρ] = w_avg * r_log

        p = r_avg / (FT(2) * b_avg)
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

        half = FT(1) / FT(2)
        γm1 = _γ(FT) - FT(1)
        c = (FT(1) / (FT(2) * γm1 * b_log) - usq_avg * half + Φ_avg)

        c1 = H[1, _ρu] * u_avg + H[1, _ρv] * v_avg + H[1, _ρw] * w_avg
        c2 = H[2, _ρu] * u_avg + H[2, _ρv] * v_avg + H[2, _ρw] * w_avg
        c3 = H[3, _ρu] * u_avg + H[3, _ρv] * v_avg + H[3, _ρw] * w_avg
        H[1, _ρe] = c * H[1, _ρ] + c1
        H[2, _ρe] = c * H[2, _ρ] + c2
        H[3, _ρe] = c * H[3, _ρ] + c3

        H[1, _ρu] += D
        H[2, _ρv] += D
        H[3, _ρw] += D


        # Compute entropy variables for state 1 eq (18)
        s1 = log((p1) / (ρ1^_γ(FT)))
        b1 = ρ1 / (FT(2) * p1)
        v1_ρ = ((_γ(FT) - s1) / γm1) - (u1^2 + v1^2 + w1^2 - FT(2) * Φ1) * b1
        v1_ρu = FT(2) * b1 * u1
        v1_ρv = FT(2) * b1 * v1
        v1_ρw = FT(2) * b1 * w1
        v1_ρe = -FT(2) * b1

        # Compute entropy variables for state 2 eq (18)
        s2 = log((p2) / (ρ2^_γ(FT)))
        b2 = ρ2 / (FT(2) * p2)
        v2_ρ = ((_γ(FT) - s2) / γm1) - (u2^2 + v2^2 + w2^2 - FT(2) * Φ2) * b2
        v2_ρu = FT(2) * b2 * u2
        v2_ρv = FT(2) * b2 * v2
        v2_ρw = FT(2) * b2 * w2
        v2_ρe = -FT(2) * b2

        Δv_ρ = v2_ρ - v1_ρ
        Δv_ρu = v2_ρu - v1_ρu
        Δv_ρv = v2_ρv - v1_ρv
        Δv_ρw = v2_ρw - v1_ρw
        Δv_ρe = v2_ρe - v1_ρe

        u_avg_n = u_avg * n⃗[1] + v_avg * n⃗[2] + w_avg * n⃗[3]
        p_avg = r_avg / (FT(2) * b_avg)
        u²_bar = FT(2) * (u_avg * u_avg + v_avg * v_avg + w_avg * w_avg) - usq_avg
        h_bar = _γ(FT) / (FT(2) * b_log * γm1) + u²_bar * half + Φ_avg
        c_bar = sqrt(_γ(FT) * p_avg / r_log)

        u_t1 = u_avg - u_avg_n * n⃗[1]
        u_t2 = v_avg - u_avg_n * n⃗[2]
        u_t3 = w_avg - u_avg_n * n⃗[3]

        λ1 = abs(u_avg_n - c_bar) * r_log / (FT(2) * _γ(FT))
        λ2 = abs(u_avg_n) * r_log * γm1 / _γ(FT)
        λ3 = abs(u_avg_n + c_bar) * r_log / (FT(2) * _γ(FT))
        λ4 = abs(u_avg_n) * p_avg

        umc1 = u_avg - c_bar * n⃗[1]
        umc2 = v_avg - c_bar * n⃗[2]
        umc3 = w_avg - c_bar * n⃗[3]

        upc1 = u_avg + c_bar * n⃗[1]
        upc2 = v_avg + c_bar * n⃗[2]
        upc3 = w_avg + c_bar * n⃗[3]

        Δv_ρu_n = Δv_ρu * n⃗[1] + Δv_ρv * n⃗[2] + Δv_ρw * n⃗[3]

        # A.13 - A.16
        w1 =
            λ1 * (
                Δv_ρ +
                (umc1 * Δv_ρu + umc2 * Δv_ρv + umc3 * Δv_ρw) +
                (h_bar - c_bar * u_avg_n) * Δv_ρe
            )
        w2 =
            λ2 * (
                Δv_ρ +
                (u_avg * Δv_ρu + v_avg * Δv_ρv + w_avg * Δv_ρw) +
                (u²_bar * half + Φ_avg) * Δv_ρe
            )
        w3 =
            λ3 * (
                Δv_ρ +
                (upc1 * Δv_ρu + upc2 * Δv_ρv + upc3 * Δv_ρw) +
                (h_bar + c_bar * u_avg_n) * Δv_ρe
            )

        Dρ = w1 + w2 + w3

        Dρu =
            w1 * umc1 + w2 * u_avg + w3 * upc1 + λ4 * (Δv_ρu - Δv_ρu_n * n⃗[1] + Δv_ρe * u_t1)
        Dρv =
            w1 * umc2 + w2 * v_avg + w3 * upc2 + λ4 * (Δv_ρv - Δv_ρu_n * n⃗[2] + Δv_ρe * u_t2)
        Dρw =
            w1 * umc3 + w2 * w_avg + w3 * upc3 + λ4 * (Δv_ρw - Δv_ρu_n * n⃗[3] + Δv_ρe * u_t3)
        Dρe = (
            w1 * (h_bar - c_bar * u_avg_n) +
            w2 * (u²_bar * half + Φ_avg) +
            w3 * (h_bar + c_bar * u_avg_n) +
            λ4 * (
                (u_t1 * Δv_ρu + u_t2 * Δv_ρv + u_t3 * Δv_ρw) +
                Δv_ρe * ((u_avg*u_avg + v_avg*v_avg + w_avg*w_avg) - u_avg_n * u_avg_n)
            )
        )

        H[1, _ρ] -= Dρ * n⃗[1] * half
        H[2, _ρ] -= Dρ * n⃗[2] * half
        H[3, _ρ] -= Dρ * n⃗[3] * half

        H[1, _ρu] -= Dρu * n⃗[1] * half
        H[2, _ρu] -= Dρu * n⃗[2] * half
        H[3, _ρu] -= Dρu * n⃗[3] * half

        H[1, _ρv] -= Dρv * n⃗[1] * half
        H[2, _ρv] -= Dρv * n⃗[2] * half
        H[3, _ρv] -= Dρv * n⃗[3] * half

        H[1, _ρw] -= Dρw * n⃗[1] * half
        H[2, _ρw] -= Dρw * n⃗[2] * half
        H[3, _ρw] -= Dρw * n⃗[3] * half

        H[1, _ρe] -= Dρe * n⃗[1] * half
        H[2, _ρe] -= Dρe * n⃗[2] * half
        H[3, _ρe] -= Dρe * n⃗[3] * half

    end
end

@inline function volume_flux(H, q1, aux1, q2, aux2)
    FT = eltype(H)

    @inbounds begin
        ρ1 = q1[_ρ]
        ρu1 = q1[_ρu]
        ρv1 = q1[_ρv]
        ρw1 = q1[_ρw]
        ρe1 = q1[_ρe]
        Φ1 = aux1[_Φ]

        u1 = ρu1 / ρ1
        v1 = ρv1 / ρ1
        w1 = ρw1 / ρ1
        p1 = pressure(ρ1, ρu1, ρv1, ρw1, ρe1, Φ1)
        b1 = ρ1 / (FT(2) * p1)
        usq1 = u1 * u1 + v1 * v1 + w1 * w1

        ρ2 = q2[_ρ]
        ρu2 = q2[_ρu]
        ρv2 = q2[_ρv]
        ρw2 = q2[_ρw]
        ρe2 = q2[_ρe]
        Φ2 = aux2[_Φ]

        u2 = ρu2 / ρ2
        v2 = ρv2 / ρ2
        w2 = ρw2 / ρ2
        p2 = pressure(ρ2, ρu2, ρv2, ρw2, ρe2, Φ2)
        b2 = ρ2 / (FT(2) * p2)
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
        alpha = b_avg * r_log / (FT(2) * b1)

        H[1, _ρ] = u_avg * r_log
        H[2, _ρ] = v_avg * r_log
        H[3, _ρ] = w_avg * r_log

        p = r_avg / (FT(2) * b_avg)
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

        γm1 = _γ(FT) - FT(1)
        half = FT(1) / FT(2)
        c = (FT(1) / (FT(2) * γm1 * b_log) - usq_avg * half + Φ_avg)

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
