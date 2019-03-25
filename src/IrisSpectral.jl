module IrisSpectral

using IrisBase,
ArnoldiMethod,
ArnoldiMethodWrapper,
LinearAlgebra,
SparseArrays,
NonlinearEigenproblems,
Distributed,
LinearAlgebra,
Random,
ProgressMeter

export resonance_eigenproblem,
cf_eigenproblem,
resonance_nonlinear_eigenproblem,
eig_kl,
eig_cf,
eig_knl


include("linear.jl")
include("nonlinear.jl")
include("contour_beyn_progress_bar.jl")


"""
    eig_kl(sim, k[, ka=0, kb=0]; display=false, kwargs...) -> k,ψ
"""
function eig_kl(sim::Simulation, k::Number, ka::Number=0, kb::Number=0; display::Bool=false, kwargs...)
    A, B, σ = resonance_eigenproblem(sim, k, ka, kb)
    decomp, history = partialschur(A, B, σ; diag_inv_B=true, kwargs...)
    @assert history.converged history
    display ? println(history) : nothing

    decomp.eigenvalues[:] = sqrt.(decomp.eigenvalues[:])
    # Normalize wavefunctions according to (ψ₁,ψ₂)=δ₁₂, which requires transformed ε or F
    normalize!(sim,decomp.Q,B)
    return decomp.eigenvalues, Array(decomp.Q)
end


"""
    eig_cf(sim, k[, ka=0, kb=0]; display=false, kwargs...) -> k,ψ
"""
function eig_cf(sim::Simulation, k::Number, ka::Number=0, kb::Number=0; η::Number=0, display::Bool=false, kwargs...)
    A, B, σ = cf_eigenproblem(sim, k, ka, kb)

    decomp, history = partialschur(A, B, η; diag_inv_B=true, kwargs...)
    @assert history.converged history
    display ? println(history) : nothing

    # Normalize wavefunctions according to (ψ₁,ψ₂)=δ₁₂, which requires transformed ε or F
    normalize!(sim,decomp.Q,B)
    return decomp.eigenvalues, Array(decomp.Q)
end


"""
    eig_knl(sim, k, ka=0, kb=0; method=contour_beyn, nk=3, display=false, quad_n=100, kwargs...) -> k,ψ
"""
function eig_knl(sim::Simulation, k::Number, ka::Number=0, kb::Number=0;
        quad_n::Int=100,
        display::Bool=false,
        method::Function=contour_beyn,
        nev::Int=3,
        quad_method=nprocs()>1 ? :ptrapz_parallel : :ptrapz,
        kwargs...
        )

    nep = resonance_nonlinear_eigenproblem(sim, k, ka, kb; check_consistency=false)
    displaylevel = display ? 1 : 0
    if display && method==contour_beyn
        k, ψ = contour_beyn(nep, true; N=quad_n, σ=k, quad_method=quad_method, neigs=nev, kwargs...)
    else
        k, ψ = method(nep; N=quad_n, σ=k, displaylevel=displaylevel, neigs=nev, quad_method=quad_method, kwargs...)
    end
    return k, ψ
end


function LinearAlgebra.normalize!(sim::Simulation,ψ,B)
    dx = sim.dis.dx
    for i ∈ 1:size(ψ,2)
        𝒩² = sum((ψ[:,i].^2).*diag(B))*(isinf(dx[1]) ? 1 : dx[1])*(isinf(dx[2]) ? 1 : dx[2])
        ψ[:,i] /= sqrt(𝒩²)*exp(complex(0,angle(ψ[end÷2-1,i])))
    end
    return nothing
end

end # module
