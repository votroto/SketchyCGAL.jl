# This file is a part of Julia. License is MIT: https://julialang.org/license
# Modified part of Julia:

for (T, saupd_name, seupd_name) in
    ((:Float64, :dsaupd_, :dseupd_),
    (:Float32, :ssaupd_, :sseupd_))
    @eval begin
        function saupdg(ido, bmat, n, which, nev, TOL::Ref{$T}, resid::Vector{$T}, ncv, v::Matrix{$T}, ldv,
            iparam, ipntr, workd::Vector{$T}, workl::Vector{$T}, lworkl, info)
            ccall(($(string(saupd_name)), libarpack), Cvoid,
                (Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                    Ref{$T}, Ref{$T}, Ref{BlasInt}, Ref{$T}, Ref{BlasInt},
                    Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ref{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                ido, bmat, n, which, nev,
                TOL, resid, ncv, v, ldv,
                iparam, ipntr, workd, workl, lworkl, info, 1, 2)
        end

        function seupdg(rvec, howmny, select, d, z, ldz, sigma,
            bmat, n, evtype, nev, TOL::Ref{$T}, resid::Vector{$T}, ncv, v::Matrix{$T}, ldv,
            iparam, ipntr, workd::Vector{$T}, workl::Vector{$T}, lworkl, info)
            ccall(($(string(seupd_name)), libarpack), Cvoid,
                (Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt}, Ref{$T}, Ref{$T}, Ref{BlasInt},
                    Ref{$T}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                    Ref{$T}, Ref{$T}, Ref{BlasInt}, Ref{$T}, Ref{BlasInt},
                    Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ref{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong, Clong),
                rvec, howmny, select, d, z, ldz,
                sigma, bmat, n, evtype, nev,
                TOL, resid, ncv, v, ldv,
                iparam, ipntr, workd, workl, lworkl, info, 1, 1, 2)
        end
    end
end