using SparseArrays

"""
	mix(ws, xs)

Linearly combine `xs` weighted by `ws`. A "generalized dot product."
"""
mix(ws, xs; kwargs...) = mapreduce(*, +, ws, xs; kwargs...)

# A slightly less naive version for sparse matrices
function mix(ws::AbstractVector{W}, xs::AbstractVector{<:AbstractSparseMatrix{X,I}}; init) where {W,X,I}
	is = Vector{I}()
	js = Vector{I}()
	vs = Vector{promote_type(W, X)}()

	n = sum(nnz, xs)
	sizehint!(is, n)
	sizehint!(js, n)
	sizehint!(vs, n)

	for (w, x) in zip(ws, xs)
		ii, ji, vi = findnz(x)
		append!(is, ii)
		append!(js, ji)
		append!(vs, vi * w)
	end

	init + sparse(is, js, vs, size(init)...)
end