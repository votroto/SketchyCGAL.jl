using LinearAlgebra: checksquare
using ProgressMeter

include("eig.jl")
include("mix.jl")
include("nystrom.jl")

dual_step_size(b, z, β, η, t) = min(4β * sqrt(t + 2) * η^2 / norm(z - b)^2, β)

"""Partial derivative of the augmented Lagrangian with respect to X."""
lagrangian_dx!(Ldx, C, As, b, y, z, β) = mix!(Ldx, y + β * (z - b), As, C)

"""Partial derivative of the augmented Lagrangian with respect to y."""
lagrangian_dy(b, z) = z - b

"""Projection step to enforce tr(X) = 1."""
correct_trace!(Λ, R) = Λ .= Λ + (1 - tr(Λ)) * I / R

"""
	sketchy_cgal(C, As, b; R, iterations=1e3, β=1, info_io=stdout)

Solve a trace constrained SDP:

	min.	⟨C, X⟩
	s.t.	⟨As[i], X⟩ <= b[i], ∀i;
			tr(X) = 1;
			X ⪰ 0.

where `C, As[1], ..., As[m]` are symmetric matrices of size `n` and `b` is a
vector of length `m`. Matrices should be scaled such that their norm is 1.
"""
function sketchy_cgal(C, As, b; R, iterations=1e3, β=1, info_io=stderr)
	progress_bar = Progress(Int(iterations), output=info_io)

	n = checksquare(C)
	m = length(b)

	sketch = Nystrom{Float64}(n, R)
	Ldx = similar(C)
	z = zeros(m)
	y = zeros(m)
	v = normalize(randn(n))

	for t in 1:iterations
		βt = β * sqrt(t + 1)
		η = 2 / (t + 1)

		approx_eigmin!(v, lagrangian_dx!(Ldx, C, As, b, y, z, βt))
		z .= z * (1 - η) + η * dot.(Ref(v), As, Ref(v))
		y .+= dual_step_size(b, z, β, η, t) * lagrangian_dy(b, z)
		update!(sketch, v, η)

		next!(progress_bar)
	end

	U, Λ = reconstruct(sketch)
	correct_trace!(Λ, R)

	U * Λ * U'
end