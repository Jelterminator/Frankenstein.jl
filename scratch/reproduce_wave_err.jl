using ForwardDiff, SparseArrays, LinearAlgebra

function diagnose_wave()
    N_w = 20
    dx_w = 1.0 / (N_w + 1)
    
    function wave!(du, u, p, t)
        v = view(u, 1:N_w^2)
        w = view(u, N_w^2+1:2*N_w^2)
        dv = view(du, 1:N_w^2)
        dw = view(du, N_w^2+1:2*N_w^2)
        dv .= w
        for j in 1:N_w, i in 1:N_w
            idx = (j - 1) * N_w + i
            l = i > 1 ? v[idx-1] : 0.0
            r = i < N_w ? v[idx+1] : 0.0
            d = j > 1 ? v[idx-N_w] : 0.0
            u_p = j < N_w ? v[idx+N_w] : 0.0
            dw[idx] = (l + r + d + u_p - 4 * v[idx]) / dx_w^2
        end
    end

    u0 = rand(2 * N_w^2)
    p = nothing
    t = 0.0
    out = similar(u0)

    println("Testing In-Place Differentiation...")
    try
        ForwardDiff.jacobian((out, x) -> wave!(out, x, p, t), out, u0)
        println("Success!")
    catch e
        println("CAUGHT ERROR: ", e)
        # Show stacktrace
        Base.display_error(e, catch_backtrace())
    end
end

diagnose_wave()
