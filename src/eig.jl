using LinearAlgebra
using IterativeSolvers: lobpcg
using SparseArrays: AbstractSparseMatrix

"""
	approx_eigmin(M, iters)

Return the approximate smallest eigenvalue and eigenvector of a matrix
`M` in `iters` iterations.
"""
function approx_eigmin(M)
	ls, vs = eigen(M, 1:1)
	only(ls), vec(vs)
end

function approx_eigmin(M::AbstractSparseMatrix)
	res = lobpcg(M, false, 1, tol=1e-3)
	res.λ, res.X
end