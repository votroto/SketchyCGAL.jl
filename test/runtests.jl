using SketchyCGAL
using Test

include("concrete.jl")
@testset "concrete examples" begin
	concrete_sanity()
	concrete_tiny()
end

include("maxcut.jl")
@testset "maxcut relaxations" begin
	maxcut_example_1()
	maxcut_example_2()
end