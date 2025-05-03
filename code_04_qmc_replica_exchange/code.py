import argparse
import numpy as np
import matplotlib.pyplot as plt

# --- Helper functions --- #
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
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            i = site_id(x, y, 0, Lx); j = site_id(x, y, 1, Lx)
            bonds.append((i, j, "x", Jx))
            if x < Lx-1:
                i = site_id(x+1, y, 0, Lx); j = site_id(x, y, 1, Lx)
                bonds.append((i, j, "y", Jy))
            else:
                if y < Ly-1:
                    i = site_id(0, y+1, 0, Lx); j = site_id(x, y, 1, Lx)
                    bonds.append((i, j, "y", Jy))
            i = site_id(x, y, 0, Lx); j = site_id(x, (y+1)%Ly, 1, Lx)
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
    epsilon = eigvals[:eigvals.shape[0]//2]
    x = 0.5 * beta * epsilon
    ax = np.abs(x)
    emx = np.exp(-ax)
    em2x = np.exp(-2*ax)
    F = - 1.0/beta * np.sum(ax + np.log1p(em2x))
    E = - 1.0/beta * np.sum(x * np.tanh(x))
    dEdbeta = - 1.0/beta**2 * np.sum(x**2 * (2.0 * emx/(1.0+em2x))**2)
    return F, E, dEdbeta

# --- Replica Exchange QMC --- #
class Replica:
    def __init__(self, beta, Lx, Ly, bonds, nbondz):
        self.beta = beta
        self.Lx = Lx; self.Ly = Ly; self.bonds = bonds; self.nbondz = nbondz
        self.eta = np.random.choice([-1, 1], size=self.nbondz)
        H = make_one_body_hamiltonian(Lx, Ly, bonds, self.eta)
        self.F, self.E, self.dEdbeta = compute_fermion_free_energy(H, beta)
        self.E_samples = []
        self.dEdbeta_samples = []

    def local_update(self):
        idx = np.random.randint(self.nbondz)
        self.eta[idx] *= -1
        H_new = make_one_body_hamiltonian(self.Lx, self.Ly, self.bonds, self.eta)
        F_new, E_new, dE_new = compute_fermion_free_energy(H_new, self.beta)
        logP = -self.beta * (F_new - self.F)
        if logP >= 0 or np.log(np.random.rand()) < logP:
            self.F, self.E, self.dEdbeta = F_new, E_new, dE_new
        else:
            self.eta[idx] *= -1

    def measure(self):
        self.E_samples.append(self.E)
        self.dEdbeta_samples.append(self.dEdbeta)

    def compute_C(self):
        E_arr = np.array(self.E_samples)
        E_mean = E_arr.mean()
        E2_mean = (E_arr**2).mean()
        dE_mean = np.mean(self.dEdbeta_samples)
        return self.beta**2 * (E2_mean - E_mean**2 - dE_mean)


def run_qmc_replica_exchange(betas, Lx, Ly, bonds, nbondz,
                              N_thermal=1000, N_measure=10000,
                              swap_interval=100):
    R = len(betas)
    replicas = [Replica(beta, Lx, Ly, bonds, nbondz) for beta in betas]
    total_steps = N_thermal + N_measure
    for step in range(total_steps):
        print("step",step)
        # Replica exchange
        if step % swap_interval == 0:
            print("step replica exchange",step)
            for i in range(R-1):
                rep_i, rep_j = replicas[i], replicas[i+1]
                # compute exchange prob f
                beta_i, beta_j = rep_i.beta, rep_j.beta
                F_i_i, F_j_j = rep_i.F, rep_j.F # F=fermion free energy
                # cross energies
                F_i_j, E_i_j, dEdbeta_i_j = compute_fermion_free_energy(
                    make_one_body_hamiltonian(Lx, Ly, bonds, rep_j.eta), beta_i)
                F_j_i, E_j_i, dEdbeta_j_i = compute_fermion_free_energy(
                    make_one_body_hamiltonian(Lx, Ly, bonds, rep_i.eta), beta_j)
                exponent = (- beta_i * F_i_j - beta_j * F_j_i
                            + beta_i * F_i_i + beta_j * F_j_j)
                # note: F_new labels correspond to F_f(...)
                f = np.exp(exponent)
                if np.random.rand() < min(1, f):
                    # swap configurations
                    eta_tmp = rep_i.eta.copy()
                    rep_i.eta, rep_j.eta = rep_j.eta, eta_tmp
                    # swap stored F, E, dEdbeta
#                    rep_i.F, rep_i.E, rep_i.dEdbeta = compute_fermion_free_energy(
#                        make_one_body_hamiltonian(Lx, Ly, bonds, rep_i.eta), beta_i)
#                    rep_j.F, rep_j.E, rep_j.dEdbeta = compute_fermion_free_energy(
#                        make_one_body_hamiltonian(Lx, Ly, bonds, rep_j.eta), beta_j)
                    rep_i.F, rep_j.F = F_i_j, F_j_i
                    rep_i.E, rep_j.E = E_i_j, E_j_i
                    rep_i.dEdbeta, rep_j.dEdbeta = dEdbeta_i_j, dEdbeta_j_i
        # Local updates
        for rep in replicas:
            for _ in range(2*Lx*Ly):
                rep.local_update()
        # Measurements
        if step >= N_thermal:
            for rep in replicas:
                rep.measure()
    # compute C for each beta
    return np.array([rep.compute_C() for rep in replicas])

# --- Main function --- #
def main():
    parser = argparse.ArgumentParser(
        description='Replica Exchange QMC for Kitaev cluster specific heat')
    parser.add_argument('--L', type=int, default=2,
                        help='Number of unit cells (Lx=Ly=L)')
    parser.add_argument('--Tmin', type=float, default=0.01, help='Minimum temperature')
    parser.add_argument('--Tmax', type=float, default=10.0, help='Maximum temperature')
    parser.add_argument('--nT', type=int, default=100, help='Number of temperature points')
    parser.add_argument('--swap', type=int, default=1, help='Replica exchange interval (steps)')
    args = parser.parse_args()

    temps = np.logspace(np.log10(args.Tmin), np.log10(args.Tmax), args.nT)[::-1]
    betas = 1.0/temps
    Lx = Ly = args.L
    bonds = make_bonds(Lx, Ly)
    nbondz = np.sum([b[2]=='z' for b in bonds])

    C_array = run_qmc_replica_exchange(
        betas, Lx, Ly, bonds, nbondz,
        N_thermal=1000, N_measure=10000,
        swap_interval=args.swap)

    # Save & plot
    np.savetxt('dat_specheat_rex', np.column_stack((temps, C_array/(2*Lx*Ly))))
    plt.figure()
    plt.plot(temps, C_array/(2*args.L**2), '-')  
    plt.xlim(args.Tmin, args.Tmax)
    plt.ylim(0, 0.4)
    plt.xscale('log'); plt.yscale('linear')
    plt.xlabel('$T$'); plt.ylabel('$C_v$')
    plt.grid(); plt.savefig('fig_specheat_rex.pdf', bbox_inches='tight')

    plt.figure()
    plt.plot(temps, C_array/(2*args.L**2), '-')  
    plt.xlim(args.Tmin, args.Tmax)
    plt.ylim(0.001, 1)
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('$T$'); plt.ylabel('$C_v$')
    plt.grid(); plt.savefig('fig_specheat_rex_log.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()

