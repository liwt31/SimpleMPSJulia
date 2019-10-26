# SimpleMPSJulia
# Density matrix renormalization group (DMRG) in matrix product state (MPS)

# This file contains the definition of matrix product state
# For theoretical backgrounds, see the [reference]:
# Ulrich Schollwöck, The density-matrix renormalization group in the age of matrix product states,
# Annals of Physics, 326 (2011), 96-192


module SimpleMPS

export MatrixProductOperator, MatrixProductState, search_groundstate, expectation

using TensorOperations
using LinearAlgebra
using LinearMaps
using IterativeSolvers
using Statistics

"""
Base class for MPS and MPO
"""
abstract type MatrixProduct end


Base.length(mp::MatrixProduct) = length(mp.mp)
Base.iterate(mp::MatrixProduct, state=1) = iterate(mp.mp, state)
Base.show(io::IO, mp::MatrixProduct) = print(io, [size(m) for m in mp.mp])


struct MatrixProductOperator <: MatrixProduct
    mp::Array{Array}
    MatrixProductOperator(mp) = new(mp)
    function MatrixProductOperator(single_mo::Array, site_num::Int)
        # the first MPO site, only contains the last row
        mpo_1 = single_mo[end:end, :, :, :]
        # the last MPO site, only contains the first column
        mpo_n = single_mo[:, :, :, 1:1]
        new([[mpo_1]; fill(single_mo, site_num - 2); [mpo_n]])
    end
end


struct MatrixProductState <: MatrixProduct
    mp::Array{Array}
    MatrixProductState(mp) = new(mp)
    """
    Construct MPS with random matrices.
    The physical dimension is obtained from `mpo` and bond dimension is set to `m`
    """
    function MatrixProductState(mpo::MatrixProductOperator, m::Int)
        bond_dim = [[1]; fill(m, length(mpo) - 1); [1]]
        pdim = [size(m)[2] for m in mpo]
        # Firstly randomly fill in the matrices
        mp = [rand(bond_dim[i], pdim[i], bond_dim[i+1]) for i in 1:length(mpo)]
        mps = new(mp)
        # Then perform canonicalisation
        for i in 1:length(mpo)-1
            push_center!(mps, i)
        end
        return mps
    end
end

"""
Flip MPO and MPS. After the operation the left side of MPO (MPS) becomes the
right side and vice versa.
Note that <mps|mpo|mps> conserves after the operation.
"""
Base.reverse(mpo::MatrixProductOperator) = MatrixProductOperator(reverse([permutedims(m, (4, 2, 3, 1)) for m in mpo]))
Base.reverse(mps::MatrixProductState) = MatrixProductState(reverse([permutedims(m, (3, 2, 1)) for m in mps]))


"""
Perform left-canonicalisation on the `idx`th site of `mps` so that the canonical center is moved to `idx+1`.
The matrices are normalized by the way.
"""
function push_center!(mps::MatrixProductState, idx::Int)
    orig_shape = size(mps.mp[idx])
    ms = reshape(mps.mp[idx], (prod(orig_shape[1:2]), :))
    f = qr(ms)
    mps.mp[idx] = reshape(Array(f.Q), orig_shape[1], orig_shape[2], :)
    @tensor mps.mp[idx+1][a, c, d] := f.R[a, b] * mps.mp[idx+1][b, c, d]
    mps.mp[idx+1] /= norm(mps.mp[idx+1])
end

"""
Calculate environment on the next site based on previous environment `res`
`ms` and `mo` are local sites of the MPS and MPO.
"""
function update_env(res, ms, mo)
    """
    Graphical notation: (* for MPS and # for MPO)
    *--a-- --e
    |     |
    |     d
    |     |
    #--b--#--g
    |     |
    |     f
    |     |
    *--c--*--h
    """
    @tensor res[e, g, h] := res[a, b, c] * conj(ms)[a, d, e] * mo[b, d, f, g] * ms[c, f, h]
end


"""
Calculate an array of left-side enrivonment tensor based on MPS and MPO.
The index `i` of the array represents the left environment of the `i`th site of the MPS.
"""
function get_env(mps::MatrixProductState, mpo::MatrixProductOperator)::Array{Array}
    sentinel = ones(1, 1, 1)
    env = [sentinel]
    for (ms, mo) in zip(mps, mpo)
        res = update_env(env[length(env)], ms, mo)
        push!(env, res)
    end
    push!(env, sentinel)
    return env
end

"""
Calculate the expectation value <mps|mpo|mps>
"""
function expectation(mps::MatrixProductState, mpo::MatrixProductOperator)::AbstractFloat
    return get_env(mps, mpo)[end-1][1][1][1]
end

"""
Sweep `mps` from left to right to find the ground state of the Hamiltonian `mpo`.
The `mps` should be right-canonical
The energies calculatd during the sweep are appended to `energies`
and `env` is pass in as the environment from the right side
"""
function optimize_oneround!(mps::MatrixProductState, mpo::MatrixProductOperator, energies::Array, env::Array)
    sentinel = ones(1, 1, 1)
    # left-side environment and right-side environment. Corresponds to $L$ and $R$ in the reference literature.
    l_env = [sentinel]
    r_env = env

    for cur_idx in 1:length(mps)
        l, mo, r = l_env[cur_idx], mpo.mp[cur_idx], r_env[end-cur_idx-1]

        orig_shape = size(mps.mp[cur_idx])
        # don't optimize on the first site because it has probably been optmized during the previous sweep
        if cur_idx != 1
            # find the lowest eigen value of the local Hamitonian operator.
            # The computation can be accelerated by using a "sparse" representation of the operator
            # because some eigen solver (such as `lobpcg` in this case) only need to know what would happen
            # if the operator is applied on a certain matrix.
            # See [power method](https://en.wikipedia.org/wiki/Power_iteration) as an example
            function h_map_f(ms)
                ms = reshape(ms, orig_shape)
                """
                Graphical notation: (* for MPS and # for MPO)
                *--a-- --h--*
                |     |     |
                |     f     |
                |     |     |
                #--b--#--g--#
                |     |     |
                |     d     |
                |     |     |
                *--c--*--e--*
                """
                @tensor res[a, f, h] := l[a, b, c] * ms[c, d, e] * mo[b, f, d, g] * r[h, g, e]
                return res
            end

            new_dim = prod(orig_shape)

            h_map = LinearMap(h_map_f, new_dim, new_dim, issymmetric=true)
            lobpcg_res = lobpcg(h_map, false, reshape(mps.mp[cur_idx], new_dim), 1)
            new_energy = lobpcg_res.λ[1]

            push!(energies, new_energy)

            mps.mp[cur_idx] = reshape(lobpcg_res.X, orig_shape)
        end

        # move the canonical center to the right side
        if cur_idx != length(mps)
            push_center!(mps, cur_idx)
        end
        # update the left-side environment
        push!(l_env, update_env(l, mps.mp[cur_idx], mpo.mp[cur_idx]))
    end

    push!(l_env, sentinel)

    return l_env
end


"""
Find the ground state of the Hamiltonian `mpo`
"""
function search_groundstate(mpo::MatrixProductOperator, bond_dimension::Int=20)
    mps = MatrixProductState(mpo, bond_dimension)
    l_env = get_env(mps, mpo)
    energies = [expectation(mps, mpo)]
    while true
        # do a right to left and then left to right sweep
        for i in 1:2
            # the mps is left canonical, so flip over to make it right canonical
            # so that we can achieve sweeping from right to left by sweeping from left to right
            mps, mpo = reverse.([mps, mpo])
            l_env = optimize_oneround!(mps, mpo, energies, l_env)
        end
        # consider it as converged if the lastest energies do not vary much
        if 10 < length(energies) && isapprox(std(energies[end-10:end]), 0, atol=1e-8)
            break
        end
    end
    return mps, energies
end

end
