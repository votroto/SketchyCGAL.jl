using SparseArrays

# It would be great if `Base.mapreduce!` was a thing.

"""
	mix!(out, ws, xs, init)

Linearly combine `xs` weighted by `ws` and store the result in `out`.
"""
mix!(out, ws, xs, init) = out .= mapreduce(*, +, ws, xs; init)

function mix!(out, ws, xs::AbstractVector{<:AbstractSparseMatrix}, init)
	map!(+, out, init)
	for i in eachindex(ws)
		for j in findall(!iszero, xs[i])
			out[j] += ws[i] * xs[i][j]
		end
	end
	out
end