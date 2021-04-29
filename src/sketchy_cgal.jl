using LinearAlgebra

include("eig.jl")
include("nystrom.jl")

"""
	mix(ws, xs)

Linearly combine `xs` weighted by `ws`.
"""
mix(ws, xs) = sum(ws .* xs)

"""
	sketchy_cgal(C, As, b; R, iterations=1e3, β=1)

Solve a trace constrained SDP:

	min		⟨C, X⟩
	s.t.	⟨As[i], X⟩ <= b[i], ∀i,
			X ∈ Symmetric PSD;
			tr(X) = 1,

where `C, As[1], ..., As[m]` are symmetric matrices of size `n`, `b` is a vector
of length `m`.
"""
function sketchy_cgal(C, As, b; R, iterations=1e3, β=1)
	n = size(C, 1)
	m = length(b)

	sketch = Nystrom{Float64}(n, R)
	z = zeros(m)
	y = zeros(m)

	for t in 1:iterations
		βt = β * sqrt(t + 1)
		qt = min(round(Int, t^0.25 * log(n)), n - 1)
		η = 2 / (t + 1)

		ξ, v = approx_eigmin(C + mix(y + βt * (z - b), As), qt)
		z = z * (1 - η) + η * dot.(As, Ref(v * v'))
		g = min(4 * β * sqrt(t + 2) * η^2 / norm(z - b)^2, β)
		y = y + g * (z - b)
		update!(sketch, v, η)
	end

	U, Λ = reconstruct(sketch)
	Λ = Λ + (1 - tr(Λ)) * I / R

	U * Λ * U'
end