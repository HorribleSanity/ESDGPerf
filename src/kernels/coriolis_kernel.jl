# Coriolis source term for f-plane (cube/channel) test cases.
#
# Applies the Coriolis acceleration to the momentum equations:
#   d(ρu)/dt += f₀ · ρv
#   d(ρv)/dt -= f₀ · ρu
#
# Based on the NUMA Fortran code (create_rhs_dynamics_volume.F90):
#   f_vec = (0, 0, f₀)  for the cube case (kvector = (0,0,1))
#   buc = -f₀·v,  bvc = f₀·u  (velocity form, then applied as -buc, -bvc)
#
# In conservative form this becomes:
#   S(ρu) = +f₀·ρv,   S(ρv) = -f₀·ρu
#
# No contribution to mass (ρ), vertical momentum (ρw), or energy (ρe),
# since the Coriolis force is perpendicular to velocity and does no work.
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
