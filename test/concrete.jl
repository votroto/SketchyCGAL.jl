using LinearAlgebra

function concrete_sanity()
	C = Symmetric(reshape([-1.0], 1, 1))
	A1 = Symmetric(reshape([1.0], 1, 1))
	b = [1.0]

	X_expected = [1]
	obj_expected = -1

	X_actual = sketchy_cgal(C, [A1], b; R=1, info_io=devnull)
	obj_actual = dot(C, X_actual)

	@test obj_actual ≈ obj_expected rtol = 1e-2
	@test X_actual ≈ X_expected rtol = 1e-2
end

function concrete_tiny()
	C = Symmetric([0.55 -0.66 0.1; -0.66 -0.44 0.52; 0.1 0.52 0.14])
	A1 = Symmetric([-0.16 0.18 -0.66; 0.18 0.34 0.46; -0.66 0.46 -0.22])
	A2 = Symmetric([0.06 -0.24 -0.41; -0.24 0.59 -0.42; -0.41 -0.42 0.54])
	b = [-0.5, 0.87]

	X_expected = [0.05 -0.03 0.08; -0.03 0.45 -0.45; 0.08 -0.45 0.51]
	obj_expected = -0.51

	X_actual = sketchy_cgal(C, [A1, A2], b; R=3, info_io=devnull)
	obj_actual = dot(C, X_actual)

	@test obj_actual ≈ obj_expected rtol = 1e-2
	@test X_actual ≈ X_expected rtol = 1e-2
end
