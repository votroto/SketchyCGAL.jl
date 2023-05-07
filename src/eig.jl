using LinearAlgebra
using SparseArrays: AbstractSparseMatrix


include("hpc_arpack.jl")
"""
	approx_eigmin!(x, M)

Return the approximate smallest eigenvalue matrix `M` starting with an initial
guess `x` and saving the corresponding eigenvector into `x`.
"""


function approx_eigmin!(x, A, w=nothing)
    dA, _v, = _eigsg(A; v0=x,workspace=w)
    x .= view(_v, :, 1)
	dA[1]
end