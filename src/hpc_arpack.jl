
# Load in our binary dependencies
using Arpack_jll
using StaticArrays

using LinearAlgebra: BlasFloat, BlasInt, Diagonal, I, SVD, UniformScaling,
    checksquare, factorize, ishermitian, issymmetric, mul!,
    rmul!, qr!
import LinearAlgebra


include("libarpackg.jl")

struct MinEigsWorkspace{T}
    workd::Vector{T}
    workl::Vector{T}
    resid::Vector{T}
    select::Vector{BlasInt}
    v::Matrix{T}
    d::Vector{T}
end

function make_min_eigs_workspace(A::AbstractMatrix{T}) where {T}
    n = checksquare(A)

    lworkl = n * (n + 8)
    workd = Vector{T}(undef, 3 * n)
    workl = Vector{T}(undef, lworkl)
    resid = Vector{T}(undef, n)

    select = Vector{BlasInt}(undef, n)
    v = Matrix{T}(undef, n, n)
    d = Vector{T}(undef, 1)

    MinEigsWorkspace{T}(workd, workl, resid, select, v, d)
end

function make_iparam(ishifts, maxiter, mode)
    iparam = @MVector zeros(BlasInt, 11)
    iparam[1] = BlasInt(ishifts)
    iparam[3] = BlasInt(maxiter)
    iparam[7] = BlasInt(mode)

    iparam
end

function _eigsg(A; workspace, v0, tol=1e-3, maxiter::Integer=300)
    n = checksquare(A)
    T = eltype(A)
    ncv = BlasInt(min(20, n))
    sigma = zero(T)
    bmat = "I"
    which = "SA"
    rvec = true
    howmny = "A"
    mode = 1
    nev = 1

    sigmar = Ref{T}(sigma)
    TOL = Ref{T}(tol)
    info = Ref{BlasInt}(1)
    ido = Ref{BlasInt}(0)

    lworkl = length(workspace.workl)
    workd = workspace.workd
    workl = workspace.workl
    select = workspace.select

    resid = workspace.resid
    v = workspace.v
    d = workspace.d
    ipntr = @MVector zeros(BlasInt, 11)
    iparam = make_iparam(1, maxiter, mode)
    
    select .*= 0
    resid[:] .= v0
    while true
        saupdg(ido, bmat, n, which, nev, TOL, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info)
        
        if ido[] == -1 || ido[] == 1
            x = view(workd, ipntr[1]:ipntr[1]+n-1)
            y = view(workd, ipntr[2]:ipntr[2]+n-1)
            mul!(y, A, x)
        else
            break
        end
    end


    seupdg(rvec, howmny, select, d, v, n, sigmar, bmat, n, which, nev, TOL, resid, ncv, v, n, iparam, ipntr, workd, workl, lworkl, info)
    if info[] != 0
        throw(XYEUPD_Exception(info[]))
    end

    return d, v

end
