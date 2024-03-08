using LinearAlgebra: checksquare
using ProgressMeter


mutable struct Lag{T} <: AbstractMatrix{T}
	C::AbstractMatrix{T}
	As::AbstractVector{AbstractMatrix{T}}
	b::AbstractVector{T}
	y::AbstractVector{T}
	z::AbstractVector{T}
	β::T
	Ldx::AbstractMatrix{T}
	function Lag{T}(C, As, b, y, z, β) where {T}
		new(C, As, b, y, z, β,similar(C))
	end
	
end

function Base.size(L::Lag)
	size(L.Ldx)
end


function LinearAlgebra.issymmetric(L::Lag)
	true
end

function LinearAlgebra.mul!(OUT, L::Lag, X)
	#lagrangian_dx!(L.Ldx, L.C, L.As, L.b, L.y, L.z, L.β)
	
	
	mul!(OUT, L.Ldx, X)
	#
	OUT .+= (L.y + L.β * (L.z - L.b)) .* X
	#@show OUT
end


include("eig.jl")
include("mix.jl")
include("nystrom.jl")

dual_step_size(b, z, β, η, t) = min(4β * sqrt(t + 2) * η^2 / norm(z - b)^2, β)

"""Partial derivative of the augmented Lagrangian with respect to X."""
lagrangian_dx!(Ldx, C, As, b, y, z, β) = mcx!(Ldx, y + β * (z - b), As, C)

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
	βt = β 

#	LL = Lag{Float64}(C, As, b, y, z, βt)
iter = LOBPCGIterator(Ldx, false, reshape(v, :, 1))
	for t in 1:iterations
		βt = β * sqrt(t + 1)
		η = 2 / (t + 1)

		lagrangian_dx!(Ldx, C, As, b, y, z, βt)
		approx_eigmin!(v, iter)
		z .= z * (1 - η) + η * dot.(Ref(v), As, Ref(v))
		y .+= dual_step_size(b, z, β, η, t) * lagrangian_dy(b, z)


		update!(sketch, v, η)

		next!(progress_bar)
	end

	U, Λ = reconstruct(sketch)
	correct_trace!(Λ, R)

	U * Λ * U'
end