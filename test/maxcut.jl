# Based on an example from
#	https://jump.dev/JuMP.jl/previews/PR2476/examples/Semidefinite%20programs/max_cut_sdp/
# which in turn uses methods from
#	Goemans, M. X., & Williamson, D. P. (1995). Improved approximation
#	algorithms for maximum cut and satisfiability problems using semidefinite
#	programming. Journal of the ACM (JACM), 42(6), 1115-1145.

using SparseArrays
using LinearAlgebra

# Model the max-cut SDP relaxation --------------------------------------------

cut_value(C, cut) =	sum(C .* (cut * cut'))
laplacian(W::Symmetric) = 0.25 * (spdiagm(vec(sum(W, dims=1))) - W)

function extract_cut(X)
	rnd = normalize(rand(size(X, 1)))

	factorization = cholesky(Hermitian(X), Val(true); check=false)
	V = (factorization.P * factorization.L)'
	vs = map(normalize, eachcol(V))

	map(x -> dot(rnd, x) |> sign, vs)
end

function max_cut_sdp_model(weights)
	n = size(weights, 1)

	As = [sparse([i], [i], [1.0], n, n) for i in 1:n]
	b = ones(n) / n

	weights, As, b
end

function solve_maxcut(W)
	n = size(W, 1)

	C, As, b = max_cut_sdp_model(laplacian(W))
	X = sketchy_cgal(normalize(-C), As, b; R=n, info_io=devnull)

	cut = extract_cut(X)
	value = cut_value(C, cut)

	cut, value
end

# Examples --------------------------------------------------------------------

function maxcut_example_1()
	W = Symmetric([
		0.0 5.0;
		5.0 0.0;
	])

	cut, value = solve_maxcut(W)

	@test cut[1] != cut[2]
end

function maxcut_example_2()
	W = Symmetric([
		0.0 5.0 7.0 6.0;
		5.0 0.0 0.0 1.0;
		7.0 0.0 0.0 1.0;
		6.0 1.0 1.0 0.0;
	])

	cut, value = solve_maxcut(W)

	@test cut[1] != cut[2]
	@test cut[2] == cut[3] == cut[4]
end

function maxcut_example_3()
	W = Symmetric([
		0.0 1.0 5.0 0.0;
		1.0 0.0 0.0 9.0;
		5.0 0.0 0.0 2.0;
		0.0 9.0 2.0 0.0;
	])

	cut, value = solve_maxcut(W)

	@test cut[1] == cut[4]
	@test cut[2] == cut[3]
	@test cut[1] != cut[2]
end