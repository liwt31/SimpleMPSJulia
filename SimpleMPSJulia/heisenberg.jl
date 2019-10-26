# SimpleMPSJulia
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the defination of Heisenberg in MPO structure
# and solves its ground state
using LinearAlgebra

include("./mps.jl")
using .SimpleMPS

"""
Construct the building block of the Heisenberg XXZ model MPO.
The returned array has 4 dimensions and the physical dimensions
are in the middle.
"""
function heisenberg_mo(J, Jz, h)::Array
    # S^+
    Sp = [[0., 1.] [0., 0.]]
    # S^-
    Sm = [[0., 0.] [1., 0.]]

    Sz = [[0.5, 0.] [0., -0.5]]

    # zero matrix block
    S0 = zeros((2, 2))

    # identity matrix block
    S1 = Diagonal(ones(2))

    mpo_block1 = cat(S1, S0, S0, S0, S0, dims=3)
    mpo_block2 = cat(Sp, S0, S0, S0, S0, dims=3)
    mpo_block3 = cat(Sm, S0, S0, S0, S0, dims=3)
    mpo_block4 = cat(Sz, S0, S0, S0, S0, dims=3)
    mpo_block5 = cat(-h * Sz, J / 2 * Sm, J / 2 * Sp, Jz * Sz, S1, dims=3)

    # after the permutation, the first index corresponds to the vertical axis of the above matrix
    # and the last index corresponds to the horizontal axis of the above matrix
    single_mo = permutedims(cat(mpo_block1, mpo_block2, mpo_block3, mpo_block4, mpo_block5, dims=4), (4, 1, 2, 3))
end

const J = Jz = h = 1
mo = heisenberg_mo(J, Jz, h)

const site_num = 20
mpo = MatrixProductOperator(mo, site_num)

const bond_dimension = 20
mps, e = search_groundstate(mpo, bond_dimension)
println("Energies:")
println(e)
