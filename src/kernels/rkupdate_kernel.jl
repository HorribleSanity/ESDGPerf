function rkupdate_knl(dq, q, rka_s1, rkb_s0, dt, ::Val{Nq3}) where {Nq3}
    ijk = threadIdx().x
    e = blockIdx().x

    id = ijk + Nq3 * nstate * (e - 0x1)

    @inbounds begin
        @unroll for s = 1:nstate
            ids = id + Nq3 * (s - 0x1)

            r_q = q[ids]
            r_dq = dq[ids]

            r_q += rkb_s0 * dt * r_dq
            r_dq *= rka_s1

            q[ids] = r_q
            dq[ids] = r_dq
        end
    end

    nothing
end
