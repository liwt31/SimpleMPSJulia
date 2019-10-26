# SimpleMPSJulia
This is a simplest demo of Matrix product state (MPS) based density matrix renormalization group (DMRG) with detailed comments, particularly suitable for new-learners to quickly get the idea of how MPS works at code level. For sophisticated and production-ready MPS packages, I recommend [pytenet](https://github.com/cmendl/pytenet) and [quimb](https://github.com/jcmgray/quimb).

---
# What is SimpleMPSJulia
`SimpleMPSJulia` aims to provide a demo for MPS based DMRG to solve the [Heisenberg model](https://en.wikipedia.org/wiki/Heisenberg_model_(quantum)).
The implementation is largely inspired by [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477v2). Understanding the first 6 parts of the article is crucial to understanding the code.

Previously I've wrote [`SimpleMPS`](https://github.com/liwt31/SimpleMPS) in Python and this project can be considered as a Julia alternative.
However the philosophy between the implementations are quite different and I'd prefer the Julia version because the code is more succint,
although it may not be as easy to understand as the Python version.

# Files
* `./SimpleMPS/mps.jl` Implements the MPS based DMRG method
* `./SimpleMPS/heisenberg.jl` Defines the hamiltonian and matrix product operator of the Heisenberg model then searches for the ground state

# How to use
* `julia heisenberg.jl` to see the energy of the system during each iteration. There are 3 parameters hard-coded in `heisenberg.jl`:
  * `J`, `Jz` and `h` which are the parameters of the model.
  * `site_num` which is the number of sites (spins) in the model.
  * `bond_dimension` which is the maximum dimension of the bond degree in matrix product state. The higher bond dimension, the higher accuracy and computational cost.
* The convergence criteria for the groundstate search is defined in `mps.jl` which is now `1e-8`.
* Modify the codes to explore DMRG and MPS!
