using Arpack
using LinearAlgebra

"""Approximates a smallest magnitude-eigenvalue and an eigenvector of a matrix 
M in `iters` iterations."""
function approx_eigmin(M, iters)
	n = size(M, 1)
	
	initial = normalize(randn(n))
	values, vecs = eigs(M, nev=1, maxiter=iters, which=:SM, v0=initial)

	values[1], vecs[:, 1]
end