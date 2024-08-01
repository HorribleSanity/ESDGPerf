function rkupdate_knl(n, dq, q, rka_s1, rkb_s0, dt)
    i = (blockIdx().x - 0x1) * blockDim().x + threadIdx().x

    @inbounds if i <= n
        r_q = q[i]
        r_dq = dq[i]

        r_q += rkb_s0 * dt * r_dq
        r_dq *= rka_s1

        q[i] = r_q
        dq[i] = r_dq
    end

    nothing
end
