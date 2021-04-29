using LinearAlgebra

mutable struct Nystrom{T}
	Ω::AbstractMatrix{T}
	S::AbstractMatrix{T}
	function Nystrom{T}(n, R) where{T}
		new(randn(T, n, R), zeros(n, R))
	end
end

function update!(sketch, v, η)
	sketch.S = (1 - η) * sketch.S + η * v * (v' * sketch.Ω)
end

function reconstruct(sketch)
	n = size(sketch.S, 1)

	σ = sqrt(n) * eps() * norm(sketch.S)
	Sσ = sketch.S + σ * sketch.Ω
	L = cholesky(Hermitian(sketch.Ω' * Sσ))
	U, Σ, = svd(Sσ / L.U)
	Λ = max.(0, Σ .^ 2 .- σ)

	U, Diagonal(Λ)
end