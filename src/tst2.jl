using LinearAlgebra

include("eig.jl")

function rand_sym_pd(n)
	a = rand(n, n)
	a' * a
end

function rand_norm_vec(n)
	a = rand(n)
	normalize(a)
end

n = 32

a = rand_sym_pd(n)
x = rand_norm_vec(n)


function ee(x, Ldx)
	v = deepcopy(x)
    w = make_min_eigs_workspace(Ldx)
	@show approx_eigmin!(v, Ldx, w)
	v
end
function ee2(x, Ldx)
	v = deepcopy(x)
	@show approx_eigmin!(v, Ldx)
	v
end

@show aa=ee(x, a)
@show bb=ee2(x, a)
