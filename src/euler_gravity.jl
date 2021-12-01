export _γ
export nstate, naux
export _ρ, _ρu, _ρv, _ρw, _ρe, _Φ

_γ(::Type{FT}) where {FT} = FT(7 / 5)

const nstate = 5
const _ρ, _ρu, _ρv, _ρw, _ρe = 1:5

const naux = 1
const _Φ = 1

@inline function boundarystate!(q⁺, aux⁺, n⃗, q⁻, aux⁻)
  FT = eltype(q⁺)
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
