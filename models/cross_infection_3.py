# Author: Karen Guzman
# Description: Two-variant SIR model with cross-infection, simplified to 7 compartments (S, I1, I2, R1, R2, S1, S2).
# Date: 7/2/2026

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# index map
S, I1, I2, R1, R2, S1, S2 = range(7)

 
def sim_cross_infection(variables, t, params):
    s, i1, i2, r1, r2, s1, s2 = variables
    beta1, beta2, gamma1, gamma2, omega = params
 
    N = s + i1 + i2 + r1 + r2 + s1 + s2
 
    # 
    dSdt  = -beta1 * s * i1 / N - beta2 * s * i2 / N          # S -> I1, S -> I2
    dI1dt =  beta1 * s * i1 / N + beta1 * s2 * i1 / N - gamma1 * i1  # (S->I1) + (S2->I1) - recovery
    dI2dt =  beta2 * s * i2 / N + beta2 * s1 * i2 / N - gamma2 * i2  # (S->I2) + (S1->I2) - recovery
    dR1dt =  gamma1 * i1 - omega * r1 # recovery -> S1
    dR2dt =  gamma2 * i2 - omega * r2 # recovery -> S2
    dS1dt =  omega * r1 - beta2 * s1 * i2 / N # S1 -> I2
    dS2dt =  omega * r2 - beta1 * s2 * i1 / N # S2 -> I1
 
    return [dSdt, dI1dt, dI2dt, dR1dt, dR2dt, dS1dt, dS2dt]
 
 
def simulator(params, y0, t):
    return integrate.odeint(sim_cross_infection, y0, t, args=(params,))

def simulate_cross_infection(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
    #Variant 1 circulates alone until t_emerge, then variant 2 is seeded by moving `seed` people from S into I2
    n1 = int(n * t_emerge / t_total)
    n2 = n - n1
 
    t1 = np.linspace(0, t_emerge, n1)
    solution1 = simulator(params, y0, t1)
 
    y_handoff = solution1[-1].copy()   # [S, I1, I2, R1, R2, S1, S2] at t_emerge
    y_handoff[S]  -= seed
    y_handoff[I2] += seed
 
    t2 = np.linspace(t_emerge, t_total, n2)
    solution2 = simulator(params, y_handoff, t2)
 
    return np.concatenate([t1, t2]), np.vstack([solution1, solution2])
 
 
def plot_cross_infection(t, y, title=None, savepath=None):
    emerge_index = np.argmax(y[:, I2] > 0)
    t_emerge = t[emerge_index]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
 
    ax1.plot(t, y[:, I1], color='tab:blue',   lw=2, label='I1 (infected, variant 1)')
    ax1.plot(t, y[:, I2], color='tab:red',    lw=2, label='I2 (infected, variant 2)')
    ax1.axvline(t_emerge, color='gray', ls='--', lw=1, label='Variant 2 emerges')
    ax1.set_ylabel('Infected')
    ax1.set_title(title or 'Epidemic curves')
    ax1.legend(loc='upper right', fontsize=9)
 
    ax2.plot(t, y[:, S],  color='black',       lw=2, label='S (never infected)')
    ax2.plot(t, y[:, R1], color='tab:blue',    ls='--', label='R1 (cross-protected, had v1)')
    ax2.plot(t, y[:, S1], color='tab:cyan',    label='S1 (waned, susceptible to v2)')
    ax2.plot(t, y[:, R2], color='tab:red',     ls='--', label='R2 (cross-protected, had v2)')
    ax2.plot(t, y[:, S2], color='tab:orange',  label='S2 (waned, susceptible to v1)')
    ax2.axvline(t_emerge, color='gray', ls='--', lw=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Population')
    ax2.legend(loc='upper right', fontsize=9)
 
    
    print(f"[{title}]")
    print(f"  Variant 1 peak: {y[:, I1].max():7.1f} at t={t[np.argmax(y[:, I1])]:.1f}")
    print(f"  Variant 2 peak: {y[:, I2].max():7.1f} at t={t[np.argmax(y[:, I2])]:.1f}")
    print(f"  Never infected at end (S): {y[-1, S]:.1f}")
 
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=110, bbox_inches='tight')
    return fig
 