import argparse
import numpy as np
import matplotlib.pyplot as plt

def make_pauli():
    """Return Pauli matrices sigma_x, sigma_y, sigma_z as numpy arrays."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}

def site_id(x, y, orb, Lx):
    return (y * Lx + x) * 2 + orb

def make_bonds(Lx, Ly, Jx=1.0/3.0, Jy=1.0/3.0, Jz=1.0/3.0):
    # Define bonds: alternate x/y within unit cells, z between cells, open boundary (omit last y-link)
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            # Bond G
            i = site_id(x, y, 0, Lx)
            j = site_id(x, y, 1, Lx)
            bonds.append((i, j, "x", Jx))
            # Bond B
            if x < Lx-1:
                i = site_id(x+1, y, 0, Lx)
                j = site_id(x, y, 1, Lx)
                bonds.append((i, j, "y", Jy))
            else: # keep for type II BC, remove for type I BC
                if y < Ly-1:
                    i = site_id(0, y+1, 0, Lx)
                    j = site_id(x, y, 1, Lx)
                    bonds.append((i, j, "y", Jy))
            # Bond R
            i = site_id(x, y, 0, Lx)
            j = site_id(x, (y+1)%Ly, 1, Lx)
            bonds.append((i, j, "z", Jz))
    return bonds

def make_hamiltonian(Lx, Ly, bonds):
    """
    Make the full Kitaev Hamiltonian on the Type-II boundary chain of 2*Lx*Ly sites.
    The 8-site cluster (2x2 unit cells) corresponds to L=4.
    """
    N = 2 * Lx * Ly
    dim = 2**N
    paulis = make_pauli()
    identity = np.eye(2, dtype=complex)
    # Construct the Hamiltonian
    H = np.zeros((dim, dim), dtype=complex)
    for i, j, btype, J in bonds:
        ops = [identity] * N
        ops[i] = paulis[btype]
        ops[j] = paulis[btype]
        H_ij = ops[0]
        for op in ops[1:]:
            H_ij = np.kron(H_ij, op)
        H -= J * H_ij
    return H

def compute_heat_capacity(eigvals, temps, k_B=1.0):
    """Compute specific heat C_v(T) from eigenvalues, using shifted energies to avoid overflow."""
    # Shift energies by minimum to improve numerical stability
    E0 = np.min(eigvals)
    energies = eigvals - E0
    beta = 1.0 / (k_B * temps)
    # Exponential terms: shape (nT, nE)
    exp_term = np.exp(-beta[:, None] * energies[None, :])
    Z = np.sum(exp_term, axis=1)
    E1 = np.sum(eigvals[None, :] * exp_term, axis=1) / Z
    E2 = np.sum(eigvals[None, :]**2 * exp_term, axis=1) / Z
    Cv = (beta**2) * (E2 - E1**2)
    return Cv

def main():
    parser = argparse.ArgumentParser(
        description='Exact diagonalization and heat capacity for Kitaev 8-site cluster (Type-II)'
    )
    parser.add_argument('--L', type=int, default=2,
                        help='Number of unit cells (use L=2 for 2x2^2-site cluster)')
    parser.add_argument('--Tmin', type=float, default=0.01, help='Minimum temperature (log scale)')
    parser.add_argument('--Tmax', type=float, default=10.0, help='Maximum temperature (log scale)')
    parser.add_argument('--nT', type=int, default=100, help='Number of temperature points (log-spaced)')
    args = parser.parse_args()
    # Make and diagonalize
    bonds = make_bonds(args.L,args.L)
    H = make_hamiltonian(args.L,args.L,bonds)
    eigvals = np.linalg.eigvalsh(H)
    np.savetxt("dat_ene",eigvals)
    # Log-spaced temperature array
    temps = np.logspace(np.log10(args.Tmin), np.log10(args.Tmax), args.nT)
    Cv = compute_heat_capacity(eigvals, temps)
    np.savetxt("dat_specheat",np.array([temps,Cv/(2*args.L**2)]).T)
    # Plot
    plt.figure()
    plt.plot(temps, Cv/(2*args.L**2),"-")
    plt.xscale("log")
    plt.xlim(args.Tmin, args.Tmax)
    plt.ylim(0, 0.4)
    plt.grid()
    plt.xlabel("$T$")
    plt.ylabel("$C_v$")
    plt.savefig("fig_specheat.pdf", bbox_inches="tight")
    plt.close()
    #
    plt.figure()
    plt.plot(temps, Cv/(2*args.L**2),"-")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(args.Tmin, args.Tmax)
    plt.ylim(0.001, 1)
    plt.grid()
    plt.xlabel("$T$")
    plt.ylabel("$C_v$")
    plt.savefig("fig_specheat_log.pdf", bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    main()

