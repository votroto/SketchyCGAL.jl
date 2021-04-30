using Arpack
using LinearAlgebra

"""
	approx_eigmin(M, iters)

Return the approximate smallest eigenvalue and eigenvector of a matrix 
`M` in `iters` iterations.
"""
function approx_eigmin(M, iters)
	values, vecs = eigs(M, nev=1, tol=1e-3, maxiter=iters, check=1)	
	values[1], vecs[:, 1]
end