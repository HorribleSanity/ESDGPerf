# f-plane Coriolis: d(ρu)/dt += f₀·ρv, d(ρv)/dt -= f₀·ρu
function coriolis_knl!(dq, q, f0, ::Val{Nq}) where {Nq}
    Nq3 = Nq^3

    e = blockIdx().x
    ijk = threadIdx().x

    offset = Nq3 * nstate * (e - 1)

    @inbounds begin
        ρu = q[ijk + Nq3 * (_ρu - 1) + offset]
        ρv = q[ijk + Nq3 * (_ρv - 1) + offset]

        dq[ijk + Nq3 * (_ρu - 1) + offset] += f0 * ρv
        dq[ijk + Nq3 * (_ρv - 1) + offset] -= f0 * ρu
    end

    nothing
end
