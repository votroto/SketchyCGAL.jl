using SparseArrays

# It would be great if `Base.mapreduce!` was a thing.

"""
	mix!(out, ws, xs, init)

Linearly combine `xs` weighted by `ws` and store the result in `out`.
"""
mix!(out, ws, xs, init) = out .= mapreduce(*, +, ws, xs; init)

function mix!(out, ws, xs::AbstractVector{<:AbstractSparseMatrix}, init)
	map!(+, out, init)
	for i in 1:length(ws)
		ii, ji, vi = findnz(xs[i])
		for j in 1:length(vi)
			out[ii[j], ji[j]] += vi[j] * ws[i]
		end
	end
	out
end