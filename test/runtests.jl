using Test

include("uniform.jl")
include("isentropicvortex.jl")
include("risingbubble.jl")
include("gravitywave.jl")

@testset "ESDGPerf tests" begin
    @testset "Uniform tests" UniformTests.runtests()
    @testset "Isentropic vortex tests" IsentropicVortexTests.runtests()
    @testset "Rising bubble tests" RisingBubbleTests.runtests()
    @testset "Gravity wave tests" GravityWaveTests.runtests()
end
