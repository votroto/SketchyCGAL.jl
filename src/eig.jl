using LinearAlgebra
using IterativeSolvers

"""
	approx_eigmin(M, iters)

Return the approximate smallest eigenvalue and eigenvector of a matrix
`M` in `iters` iterations.
"""
function approx_eigmin(M)
	res = lobpcg(M, false, 1, tol=1e-3)
	res.λ, res.X
end