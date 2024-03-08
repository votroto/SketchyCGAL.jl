using SparseArrays
using LinearAlgebra

include("sketchy_cgal.jl")

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
	scale_b = 4

	X = sketchy_cgal(-C / scale_C, As, b / scale_b; R = n, info_io = devnull)

	dot(C, X * scale_b)
end

# Examples --------------------------------------------------------------------

	W = sparse([
		0 5 7 6
		5 0 0 1
		7 0 0 1
		6 1 1 0
	])

	value = approx_max_cut_value(W)

