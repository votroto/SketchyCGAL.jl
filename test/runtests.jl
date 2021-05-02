using SketchyCGAL
using Test

include("maxcut.jl")

@testset "maxcut examples" begin
	maxcut_example_1()
	maxcut_example_2()
	maxcut_example_3()
end