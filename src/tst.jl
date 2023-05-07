using LinearAlgebra
using JuMP
using MosekTools
using Random
using SparseArrays
include("sketchy_cgal.jl")


Random.seed!(3232)

NN = 100
C = sparse(Symmetric(sprand(NN, NN, 0.1)))
A1 = sparse(Symmetric(sprand(NN, NN, 0.1)))
A2 = sparse(Symmetric(sprand(NN, NN, 0.1)))
c=C
#println(eigs(C))
#println(norm(Array(C))," ",opnorm(Array(C)))
#println(norm(Array(A1))," ",opnorm(Array(A1)))
#println(norm(Array(A2))," ",opnorm(Array(A2)))
C = C / opnorm(Matrix(C))
A1 = A1 / opnorm(Matrix(A1))
A2 = A2 / opnorm(Matrix(A2))
b = [0.5,0.5]

function ppcsdp(C,A1,A2,b)
	m = Model(optimizer_with_attributes(MosekTools.Optimizer, "MSK_IPAR_NUM_THREADS"=>1))
	@variable m X[axes(C,1), axes(C,2)]
	@objective m Min dot(C, X)
	@constraint m dot(A1, X) == b[1]
	@constraint m dot(A2, X) == b[2]
	@constraint m X ∈ PSDCone()
	@constraint m tr(X) == 1
set_silent(m)
	optimize!(m)

	#@show termination_status(m)
	@show obj_expected = objective_value(m)
	#@show X_expected = value.(X)
end


function ppcgal(C,A1,A2,b;iters)
	X_actual = sketchy_cgal(Matrix(C), [A1, A2], b; R=4, iterations=iters)
	dot(C, X_actual)
end

