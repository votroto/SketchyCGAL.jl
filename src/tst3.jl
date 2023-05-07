using LinearAlgebra
using JuMP
using MosekTools
using Random
using SparseArrays
include("sketchy_cgal.jl")


Random.seed!(3232)


# Model the max-cut SDP relaxation --------------------------------------------

laplacian(W) = spdiagm(vec(sum(W, dims = 1))) - W

function max_cut_sdp_model(weights)
	n = size(weights, 1)

	C = laplacian(weights) / 4
	b = ones(n)

	C, b
end

function cgal_relax(W)
	X, v = approx_max_cut_value(W)

	v
end

function actual_relax(W)
    L = laplacian(W)

    m = Model(optimizer_with_attributes(MosekTools.Optimizer, "MSK_IPAR_NUM_THREADS"=>1))
	@variable m X[axes(L,1), axes(L,2)]
	@objective m Max dot(L, X)/4
	@constraint m X ∈ PSDCone()
	@constraint m diag(X) .== 1
    
    set_silent(m)
	optimize!(m)

	@show obj_expected = objective_value(m)
end

R = sprand(100, 100, 0.2)
R += R'

cgal_relax(R)
#actual_relax(R)

@show @time cgal_relax(R)
#@show @time actual_relax(R)