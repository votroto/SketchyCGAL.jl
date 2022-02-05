# SketchyCGAL.jl
A Julia implementation of the SketchyCGAL SDP solver.
```
sketchy_cgal(C, As, b; R, iterations=1e3, β=1, info_io=stdout)
```

Solves a trace constrained SDP:
```
min.    ⟨C, X⟩
s.t.    ⟨As[i], X⟩ == b[i], ∀i;
        tr(X) = 1;
        X ⪰ 0.
```
where `C, As[1], ..., As[m]` are symmetric matrices of size `n` and `b` is a
vector of length `m`. Matrices should be scaled such that their spectral norm
is 1.

## Acknowledgement
This project is based on

	Yurtsever, A., Tropp, J., Fercoq, O., Udell, M., & Cevher, V. (2021). Scalable Semidefinite Programming. SIAM Journal on Mathematics of Data Science, 3(1), 171-200.