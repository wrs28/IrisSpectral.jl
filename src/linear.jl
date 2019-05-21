"""
    resonance_eigenproblem(sim, k, ka=0, kb=0) -> A,B,σ
"""
function resonance_eigenproblem(sim::Simulation, k::Number, ka::Number=0, kb::Number=0)
    N,args = sim.dis.N,(sim.dis.coordinate_system,sim.bnd.∂Ω,sim.bnd.bc,sim.bnd.bl)
    (∇²,fs,s),defs = laplacian(N,args...,k,ka,kb,0)
    A = spzeros(ComplexF64,prod(N),prod(N))
    for i ∈ eachindex(∇²)
        A += ∇²[i]*fs[i](k,ka,kb)
    end
    A = (A + SparseMatrixCSC(transpose(A)))/2
    B = s*spdiagm(0=>-sim.sys.ε[:])
    return A::SparseMatrixCSC{ComplexF64,Int}, B, k^2
end


"""
    resonance_eigenproblem(sim, k, ka=0, kb=0; η=0) -> A,B,σ
"""
function cf_eigenproblem(sim::Simulation, k::Number, ka::Number=0, kb::Number=0; η::Number=0)
    N,args = sim.dis.N,(sim.dis.coordinate_system,sim.bnd.∂Ω,sim.bnd.bc,sim.bnd.bl)
    (∇²,fs,s),defs = laplacian(N,args...,k,ka,kb,0)
    A = spzeros(eltype(∇²[1]),prod(N),prod(N))
    for i ∈ eachindex(∇²)
        A += ∇²[i]*fs[i](k,ka,kb)
    end
    A = (A + SparseMatrixCSC(transpose(A)))/2 - s*spdiagm(0=>-sim.sys.ε[:]*k^2)
    B = s*spdiagm(0=>-sim.sys.F[:]*k^2)
    return A, B, η
end
