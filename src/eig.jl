using Arpack
using LinearAlgebra

"""
	approx_eigmin(M, iters)

Return the approximate smallest eigenvalue and eigenvector of a matrix 
`M` in `iters` iterations.
"""
function approx_eigmin(M)
	values, vecs = eigs(M, nev=1, tol=1e-1, which=:SR)	
	values[1], vecs[:, 1]
end