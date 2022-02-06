using SparseArrays

"""
	muladd!(Y, X, a)

Inplace multiply-add, stores the result of `Y + X*a` into `Y`.
"""
function muladd!(Y, X, a)
	BLAS.axpy!(a, X, Y)
end

function muladd!(Y, X::AbstractSparseMatrix, a)
	nzs = nonzeros(X)
	rvs = rowvals(X)
	@inbounds for col in 1:size(X, 2), j in nzrange(X, col)
		Y[rvs[j], col] += nzs[j] * a
	end
	Y
end

"""
	mix!(out, ws, xs, init)

Linearly combine `xs` weighted by `ws` and store the result in `out`.
"""
function mix!(out, ws, xs, init)
	fill!(out, zero(eltype(out)))
	muladd!(out, init, 1)
	for i in eachindex(ws)
		muladd!(out, xs[i], ws[i])
	end
	out
end