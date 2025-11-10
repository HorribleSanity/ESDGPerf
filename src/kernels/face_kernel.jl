function face_knl(
    mask,
    dq,
    q,
    aux,
    faceix⁻,
    faceix⁺,
    sgeo,
    vgeo,
    boundaryfaces,
    ::Val{Nq},
    ::Val{faceoffsets},
) where {Nq,faceoffsets}
    FT = eltype(dq)
    Nq3 = Nq^3
    nfaces = 6

    e⁻ = mask[blockIdx().x]
    i = threadIdx().x

    l_H = MMatrix{3,nstate,FT}(undef)

    l_q⁻ = MVector{nstate,FT}(undef)
    l_q⁺ = MVector{nstate,FT}(undef)
    l_aux⁻ = MVector{naux,FT}(undef)
    l_aux⁺ = MVector{naux,FT}(undef)

    @inbounds begin
        @unroll for face = 1:nfaces
            j = i + faceoffsets[face]

            id⁻ = faceix⁻[j, e⁻]
            ijk⁻ = (id⁻ - 1) % Nq3 + 1

            boundarytag = boundaryfaces[face, e⁻]

            n⃗ = SVector(sgeo[j, 1, e⁻], sgeo[j, 2, e⁻], sgeo[j, 3, e⁻])
            fMJ = sgeo[j, 5, e⁻]
            MJI = 1 / vgeo[ijk⁻, _MJ, e⁻]

            @unroll for s = 1:nstate
                l_q⁻[s] = q[ijk⁻+Nq3*(s-1)+Nq3*nstate*(e⁻-1)]
            end
            @unroll for s = 1:naux
                l_aux⁻[s] = aux[ijk⁻+(s-1)*Nq3+Nq3*naux*(e⁻-1)]
            end

            if boundarytag == 0
                id⁺ = faceix⁺[j, e⁻]
                e⁺ = (id⁺ - 1) ÷ Nq3 + 1
                ijk⁺ = (id⁺ - 1) % Nq3 + 1

                @unroll for s = 1:nstate
                    l_q⁺[s] = q[ijk⁺+Nq3*(s-1)+Nq3*nstate*(e⁺-1)]
                end
                @unroll for s = 1:naux
                    l_aux⁺[s] = aux[ijk⁺+(s-1)*Nq3+Nq3*naux*(e⁺-1)]
                end
            else
                boundarystate!(l_q⁺, l_aux⁺, n⃗, l_q⁻, l_aux⁻)
            end

            volume_flux(l_H, l_q⁻, l_aux⁻, l_q⁺, l_aux⁺, n⃗) 


            @unroll for s = 1:nstate
                dq[ijk⁻+Nq3*(s-1)+Nq3*nstate*(e⁻-1)] -=
                    (l_H[1, s] * n⃗[1] + l_H[2, s] * n⃗[2] + l_H[3, s] * n⃗[3]) * fMJ * MJI
            end

            mod(face, 2) == 0 && sync_threads()
        end
    end

    nothing
end
