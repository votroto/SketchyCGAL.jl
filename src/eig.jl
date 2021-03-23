using Arpack
using LinearAlgebra

function approx_eigmin(M, q)
	n = size(M, 1)
	iters = min(round(Int, q), n - 1)
	
	initial = randn(size(M, 1))
	normalize!(initial)

	values, vecs = eigs(M, nev=1, maxiter=iters, which=:SM, v0=initial)
	values[1], vecs[:, 1]
end