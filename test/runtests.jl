using Test
@testset "ESDGPerf tests" begin
    include("uniform.jl")
    include("isentropicvortex.jl")
    include("risingbubble.jl")
    include("gravitywave.jl")
end
