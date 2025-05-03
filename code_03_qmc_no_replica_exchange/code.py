import argparse
import numpy as np
import matplotlib.pyplot as plt

def site_id(x, y, orb, Lx):
    return (y * Lx + x) * 2 + orb

def make_etas(N):
    Nstate = 2**N
    states = np.zeros((Nstate,N),dtype=int)
    for i in range(Nstate):
        tmp = i
        for j in range(N):
            states[i,N-j-1] = tmp%2
            tmp = tmp//2
    states = 1 - 2*states
    return states
    
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

def make_one_body_hamiltonian(Lx, Ly, bonds, eta):
    N = 2 * Lx * Ly
    H = np.zeros((N, N), dtype=complex)
    cnt = 0
    for i, j, btype, J in bonds:
        if btype=="z":
            sgn = eta[cnt]
            H[i,j] = -2j * J * sgn
            H[j,i] = +2j * J * sgn
            cnt += 1
        else:
            H[i,j] = -2j * J
            H[j,i] = +2j * J
    return H

def compute_fermion_free_energy(H, beta):
    eigvals = np.linalg.eigvalsh(H)
    # keep only half of eigenvalues
    epsilon = eigvals[:eigvals.shape[0]//2]
    # x = beta epsilon / 2
    x = 0.5 * beta * epsilon
    ax = np.abs(x)
    emx = np.exp(-ax)
    em2x = np.exp(-2*ax)
    # F/T = - sum log[2 cosh(x)] in stable form: |x| + log1p(exp(-2|x|))
    F = - 1.0/beta * np.sum(ax + np.log1p(em2x))
    # E/T = - sum x tanh(x)
    E = - 1.0/beta * np.sum(x * np.tanh(x))
    # d E / d beta / T^2 = - sum x^2 / cosh^2(x)
    dEdbeta = - 1.0/beta**2 * np.sum(x**2 * (2.0 * emx/(1.0+em2x))**2)
    return F, E, dEdbeta

def run_qmc(beta, Lx, Ly, bonds, nbondz, N_thermal=400, N_measure=800):
    eta = np.random.choice([-1, 1], size=nbondz)
    H = make_one_body_hamiltonian(Lx, Ly, bonds, eta)
    F, E, dEdbeta = compute_fermion_free_energy(H, beta)
    E_samples = []
    dEdbeta_samples = []
    for step in range(N_thermal + N_measure):
        for _ in range(2*Lx*Ly):
            idx = np.random.randint(nbondz)
            eta[idx] *= -1
            H_new = make_one_body_hamiltonian(Lx, Ly, bonds, eta)
            F_new, E_new, dEdbeta_new = compute_fermion_free_energy(H_new, beta)
            # Metropolis: use log-probability to avoid overflow
            logP = -beta * (F_new - F)
            if logP >= 0 or np.log(np.random.rand()) < logP:
                # accept
                F, E, dEdbeta = F_new, E_new, dEdbeta_new
            else:
                # reject
                eta[idx] *= -1
        if step >= N_thermal:
            E_samples.append(E)
            dEdbeta_samples.append(dEdbeta)
    E_mean = np.mean(E_samples)
    E2_mean = np.mean(np.array(E_samples)**2)
    dEdbeta_mean = np.mean(dEdbeta_samples)
    C = beta**2 * (E2_mean - E_mean**2 - dEdbeta_mean)
    return C

def main():
    parser = argparse.ArgumentParser(
        description='Heat capacity by QMC for Kitaev 8-site cluster (Type-II)'
    )
    parser.add_argument('--L', type=int, default=2,
                        help='Number of unit cells (use L=2 for 2x2^2-site cluster)')
    parser.add_argument('--Tmin', type=float, default=0.01, help='Minimum temperature (log scale)')
    parser.add_argument('--Tmax', type=float, default=10.0, help='Maximum temperature (log scale)')
    parser.add_argument('--nT', type=int, default=100, help='Number of temperature points (log-spaced)')
    args = parser.parse_args()
    # Set parameters
    temps = np.logspace(np.log10(args.Tmin), np.log10(args.Tmax), args.nT)[::-1]
    betas = 1.0/temps
    print(betas)
    Lx = args.L
    Ly = args.L
    bonds = make_bonds(Lx,Ly)
    nbondz = np.sum([b[2]=="z" for b in bonds])
    N_thermal = 1000 # thermalization steps
    N_measure = 10000 # measurement steps
    # Run QMC over beta
    Cv = []
    for beta in betas:
        print(f"Running beta = {beta:.3f}")
        Cv.append(run_qmc(beta,Lx,Ly,bonds,nbondz,N_thermal,N_measure))
    Cv = np.array(Cv)
    np.savetxt("dat_specheat",np.array([temps,Cv/(2*Lx*Ly)]).T)
    # Plot
    plt.figure()
    plt.plot(temps, Cv/(2*Lx*Ly),"-")
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
    plt.plot(temps, Cv/(2*Lx*Ly),"-")
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

