# SketchyCGAL.jl
A Julia implementation of the SketchyCGAL SDP solver.
```
sketchy_cgal(C, As, b)
```

Solves a trace constrained SDP:
```
min		⟨C, X⟩
s.t.	⟨As[i], X⟩ <= b[i], ∀i,
		X ∈ Symmetric PSD;
		tr(X) = 1,
```
where `C, As[1], ..., As[m]` are symmetric matrices of size `n` and 
`b` is a vector of length `m`. Preferably `‖C‖ = ‖As[i]‖ = 1`.

## Acknowledgement
This project is based on

	Yurtsever, A., Tropp, J., Fercoq, O., Udell, M., & Cevher, V. (2021). Scalable Semidefinite Programming. SIAM Journal on Mathematics of Data Science, 3(1), 171-200.