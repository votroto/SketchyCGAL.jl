
using LinearAlgebra
using Arpack
using IterativeSolvers
using SparseArrays: AbstractSparseMatrix
"""
	approx_eigmin!(x, M)

Return the approximate smallest eigenvalue matrix `M` starting with an initial
guess `x` and saving the corresponding eigenvector into `x`.
"""
function approx_eigmin!(x, iter::LOBPCGIterator)
	res = lobpcg!(iter, tol=1e-3)
	res.λ
end


function approx_eigmin!(x, iter)
	res = powm!(iter, x, tol=1e-3)
end


using Arpack
using LinearAlgebra

"""
	approx_eigmin(M, iters)
Return the approximate smallest eigenvalue and eigenvector of a matrix
`M` in `iters` iterations.
"""
function approx_eigmin!(x, M::Lag)
	values, vecs = eigs(M, nev=1, tol=1e-1, which=:SR)
	x .= vec(vecs)
	values[1]
end