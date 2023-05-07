using LinearAlgebra: checksquare
using ProgressMeter

include("eig.jl")
include("mix.jl")
include("nystrom.jl")

dual_step_size(b, z, β, η, t) = min(4β * sqrt(t + 2) * η^2 / norm(z - b)^2, β)

lagrangian_dx!(Ldx, C, yLdyb) = nymix!(Ldx, yLdyb, C)

lagrangian_dy!(Ldy, b, z) = Ldy .= z .- b

correct_trace!(Λ, R) = Λ .= Λ + (1 - tr(Λ)) * I / R

is_power_2(x) = (x & (x - 1)) == 0

#surrogate_duality_gap() 
function primal_variable(sketch, R; correct=true)
    U, Λ = reconstruct(sketch)
    if correct
        correct_trace!(Λ, R)
    end
    U * Λ * U'
end

laplacian(W) = spdiagm(vec(sum(W, dims=1))) - W

function max_cut_sdp_model(weights)
    n = size(weights, 1)

    C = laplacian(weights) / 4
    b = ones(n)

    C, b
end

function approx_max_cut_value(W)
    C, b = max_cut_sdp_model(W)

    n = size(W, 1)
    scale_C = opnorm(Matrix(C))
    scale_b = n

    X = ny(-C / scale_C, b / scale_b; R=10)

    X, dot(C, X * scale_b)
end

function ny(C, b; R, iterations=1000, β=1)
    n = checksquare(C)
    m = length(b)

    sketch = Nystrom{Float64}(n, R)
    z = zeros(m)
    y = zeros(m)
    v = normalize(randn(n))

    Ldx = similar(C)
    Ldy = z - b

    eig_mem = make_min_eigs_workspace(Ldx)
    vv_mem = similar(v)

    obj_track = 0

    for t in 1:iterations
        βt = β * sqrt(t + 1)
        η = 2 / (t + 1)

        mul!(Ldy, y, 1, 1, βt)
        lagrangian_dx!(Ldx, C, Ldy)
        approx_eigmin!(v, Ldx, eig_mem)

        copyto!(vv_mem, v .^ 2)
        mul!(z, vv_mem, 1, η, (1 - η))
        lagrangian_dy!(Ldy, b, z)
        step = dual_step_size(b, z, β, η, t)
        muladd!(y, Ldy, step)
        update!(sketch, v, η)


        obj_track = obj_track * (1 - η) + η * v' * C * v
        @show obj_track * n
        if t >= 32 && is_power_2(t)
        end
    end

    primal_variable(sketch, R)
end