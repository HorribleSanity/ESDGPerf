export _γ
export nstate, naux
export _ρ, _ρu, _ρv, _ρw, _ρe, _Φ
export entropyintegral

@inline _γ(::Type{FT}) where {FT} = FT(7 / 5)

const nstate = 5
const _ρ, _ρu, _ρv, _ρw, _ρe = 1:5

const naux = 1
const _Φ = 1

@inline function boundarystate!(q⁺, aux⁺, n⃗, q⁻, aux⁻)
    @inbounds begin
        aux⁺[_Φ] = aux⁻[_Φ]

        q⁺[_ρ] = q⁻[_ρ]
        q⁺[_ρe] = q⁻[_ρe]
        ρu⃗⁻ = SVector(q⁻[_ρu], q⁻[_ρv], q⁻[_ρw])
        ρu⃗⁺ = ρu⃗⁻ - 2 * (n⃗' * ρu⃗⁻) * n⃗
        q⁺[_ρu] = ρu⃗⁺[1]
        q⁺[_ρv] = ρu⃗⁺[2]
        q⁺[_ρw] = ρu⃗⁺[3]
    end
end

@inline function entropy(q, aux)
    FT = eltype(q)
    γ = _γ(FT)
    @inbounds begin
        ρ = q[_ρ]
        ρu⃗ = SVector(q[_ρu], q[_ρv], q[_ρw])
        ρe = q[_ρe]
        Φ = aux[_Φ]

        p = (γ - 1) * (ρe - ρu⃗' * ρu⃗ / 2ρ - ρ * Φ)
        s = log(p / ρ^γ)
        -ρ * s / (γ - 1)
    end
end

function entropy!(η, q, aux, ::Val{Nq}) where {Nq}
    FT = eltype(η)
    Nq3 = Nq^3

    e = blockIdx().x
    ijk = threadIdx().x

    l_q = MVector{nstate,FT}(undef)
    l_aux = MVector{naux,FT}(undef)

    @inbounds begin
        @unroll for s = 1:nstate
            l_q[s] = q[ijk+Nq3*(s-1)+Nq3*nstate*(e-1)]
        end
        @unroll for s = 1:naux
            l_aux[s] = aux[ijk+Nq3*(s-1)+Nq3*naux*(e-1)]
        end
        η[ijk+Nq3*(e-1)] = entropy(l_q, l_aux)
    end
    nothing
end

function entropyintegral(q, aux, MJ, grid)
    Nq3 = size(q, 1)
    Nq = Int(cbrt(Nq3))
    Ne = length(grid)
    η = similar(q, (Nq3, 1, Ne))
    @cuda threads = Nq3 blocks = Ne entropy!(η, q, aux, Val(Nq))
    s = sum(@. MJ * η)
    MPI.Allreduce(s, +, Raven.comm(grid))
end
