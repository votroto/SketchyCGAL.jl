using LinearAlgebra

include("eig.jl")
include("nystrom.jl")

"""linearly interpolate between xs weighted by ws"""
mix(ws, xs) = sum(w .* x for (w, x) in zip(ws, xs))

function sketchy_cgal(C, As, b; R, iterations=1e3, β=1, K=Inf)
	n = size(C, 1)
	d = length(b)

	sketch = Nystrom(n, R)
	z = zeros(d)
	y = zeros(d)
	for t in 1:iterations
		βt = β * sqrt(t + 1)
		qt = t^0.25 * log(n)
		η = 2 / (t + 1)

		ξ, v = approx_eigmin(C + mix(y + βt * (z - b), As), qt)
		z = z * (1 - n) + η * dot.(As, Ref(v * v'))
		g = min(4 * β * (t + 1)^-1.5 / norm(z - b)^2, β)
		y = y + g * (z - b)
		update!(sketch, v, η)
	end
	U, Λ = reconstruct(sketch)
	Λ = Λ + tr(Λ) * I / R
	U * Λ * U'
end