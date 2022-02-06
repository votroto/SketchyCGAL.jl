# Based on an example from
#	https://jump.dev/JuMP.jl/previews/PR2476/examples/Semidefinite%20programs/max_cut_sdp/
# which in turn uses methods from
#	Goemans, M. X., & Williamson, D. P. (1995). Improved approximation
#	algorithms for maximum cut and satisfiability problems using semidefinite
#	programming. Journal of the ACM (JACM), 42(6), 1115-1145.

using SparseArrays
using LinearAlgebra

# Model the max-cut SDP relaxation --------------------------------------------

laplacian(W) = spdiagm(vec(sum(W, dims = 1))) - W

function max_cut_sdp_model(weights)
	n = size(weights, 1)

	C = laplacian(weights) / 4
	As = [sparse([i], [i], [1], n, n) for i in 1:n]
	b = ones(n)

	C, As, b
end

function approx_max_cut_value(W)
	C, As, b = max_cut_sdp_model(W)

	n = size(W, 1)
	scale_C = opnorm(Matrix(C))
	scale_b = n

	X = sketchy_cgal(-C / scale_C, As, b / scale_b; R = n, info_io = devnull)

	dot(C, X * scale_b)
end

# Examples --------------------------------------------------------------------

function maxcut_example_1()
	W = sparse([
		0 5 7 6
		5 0 0 1
		7 0 0 1
		6 1 1 0
	])

	value = approx_max_cut_value(W)

	@test value ≈ 18 rtol = 1e-2
end

function maxcut_example_2()
	W = sparse([
		0 1 5 0
		1 0 0 9
		5 0 0 2
		0 9 2 0
	])

	value = approx_max_cut_value(W)

	@test value ≈ 17 rtol = 1e-2
end