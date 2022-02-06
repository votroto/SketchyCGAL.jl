using LinearAlgebra
using IterativeSolvers
using SparseArrays: AbstractSparseMatrix

"""
	approx_eigmin!(x, M)

Return the approximate smallest eigenvalue matrix `M` starting with an initial
guess `x` and saving the corresponding eigenvector into `x`.
"""
function approx_eigmin!(x, M)
	res = lobpcg!(LOBPCGIterator(M, false, reshape(x, :, 1)), tol=1e-3)
	res.λ
end