# Rayleigh sponge: damps ρw toward zero for z > z_s.
# α(z) = α_max · sin²(π/2 · (z - z_s) / (z_top - z_s))
function sponge_knl!(dq, q, aux, α_max, z_s, z_top, g, ::Val{Nq}) where {Nq}
    Nq3 = Nq^3

    e = blockIdx().x
    ijk = threadIdx().x

    state_offset = Nq3 * nstate * (e - 1)
    aux_offset = Nq3 * naux * (e - 1)

    @inbounds begin
        Φ = aux[ijk + Nq3 * (_Φ - 1) + aux_offset]
        z = Φ / g

        if z > z_s
            frac = (z - z_s) / (z_top - z_s)
            α = α_max * sin(oftype(frac, π) / oftype(frac, 2) * frac)^2

            ρw = q[ijk + Nq3 * (_ρw - 1) + state_offset]
            dq[ijk + Nq3 * (_ρw - 1) + state_offset] -= α * ρw
        end
    end

    nothing
end
