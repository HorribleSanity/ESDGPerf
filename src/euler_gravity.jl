export _γ
export nstate, naux
export _ρ, _ρu, _ρv, _ρw, _ρe, _Φ

_γ(::Type{FT}) where {FT} = FT(7 / 5)

const nstate = 5
const _ρ, _ρu, _ρv, _ρw, _ρe = 1:5

const naux = 1
const _Φ = 1
