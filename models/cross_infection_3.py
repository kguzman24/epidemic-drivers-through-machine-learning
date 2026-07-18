# Author: Karen Guzman
# Description: Two-variant SIR model with cross-infection, simplified to 7 compartments (S, I1, I2, R1, R2, S1, S2).
# This version includes WANING IMMUNITY.
# Date: 7/2/2026

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

# index map
S, I1, I2, R1, R2, S1, S2 = range(7)

 
def sim_cross_infection(variables, t, params):
    s, i1, i2, r1, r2, s1, s2 = variables
    beta1, beta2, gamma1, gamma2, omega = params
 
    N = s + i1 + i2 + r1 + r2 + s1 + s2
 
    dSdt  = -beta1 * s * i1 / N - beta2 * s * i2 / N          # S -> I1, S -> I2
    dI1dt =  beta1 * s * i1 / N + beta1 * s2 * i1 / N + beta1 * s1 * i1 / N + beta1 * r2 * i1 / N - gamma1 * i1  # (S->I1) + (S2->I1) + (S1->I1) + (R2->I1) - recovery
    dI2dt =  beta2 * s * i2 / N + beta2 * s1 * i2 / N + beta2 * s2 * i2 / N + beta2 * r1 * i2 / N - gamma2 * i2  # (S->I2) + (S1->I2) + (S2->I2) + (R1->I2) - recovery
    dR1dt =  gamma1 * i1 - omega * r1 - beta2 * r1 * i2 / N # recovery -> S1, recovery ->I2
    dR2dt =  gamma2 * i2 - omega * r2 - beta1 * r2 * i1 / N # recovery -> S2, recovery ->I1
    dS1dt =  omega * r1 - beta2 * s1 * i2 / N - beta1 * s1 * i1 / N # S1 -> I2, S1->I1
    dS2dt =  omega * r2 - beta1 * s2 * i1 / N - beta2 * s2 * i2 / N # S2 -> I1, S2->I2
 
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
 
 
def plot_cross_infection(t, y, params, filename = None, figures_dir="figures"):
    S, I1, I2, R1, R2, S1, S2 = range(7) # indices for the compartments
    beta1, beta2, gamma1, gamma2, omega = params
    emerge_index = np.argmax(y[:, I2] > 0)
    t_emerge = t[emerge_index]
    N = y[0].sum()  # total population

    #R0
    R0_1 = beta1 / gamma1
    R0_2 = beta2 / gamma2

    s_emerge = y[emerge_index, S]
    s1_emerge = y[emerge_index, S1]
    s2_emerge = y[emerge_index, S2]
    r1_emerge = y[emerge_index, R1]
    effective_R0_2 = R0_2 * (s_emerge + s2_emerge + r1_emerge + s1_emerge)/ N #variant 2 can infect S, S1, S2, and R1

    #plot 
    f = plt.figure(figsize=(9, 10))
    gs = f.add_gridspec(3, 2, height_ratios=[4, 4, 1.5], width_ratios=[1, 1])
    ax1 = f.add_subplot(gs[0, :])
    ax2 = f.add_subplot(gs[1, :])
    ax_text = f.add_subplot(gs[2, 0])
    ax_flags = f.add_subplot(gs[2, 1])
 
    ax1.plot(t, y[:, I1], color='tab:blue',   lw=2, label='I1 (infected, variant 1)')
    ax1.plot(t[emerge_index:], y[emerge_index:, I2], color='tab:red',    lw=2, label='I2 (infected, variant 2)')
    ax1.axvline(t_emerge, color='gray', ls='--', lw=1, label='Variant 2 emerges')
    ax1.set_ylabel('Infected')
    ax1.set_title('Epidemic curves')
    ax1.legend(loc='upper right', fontsize=9)
 
    ax2.plot(t, y[:, S],  color='black',       lw=2, label='S (never infected)')
    ax2.plot(t, y[:, R1], color='tab:blue',    ls='--', label='R1 (cross-protected, had v1)')
    ax2.plot(t, y[:, S1], color='tab:cyan',    label='S1 (waned, susceptible to v2)')
    ax2.plot(t[emerge_index:], y[emerge_index:, R2], color='tab:red',     ls='--', label='R2 (cross-protected, had v2)')
    ax2.plot(t[emerge_index:], y[emerge_index:, S2], color='tab:orange',  label='S2 (waned, susceptible to v1)')
    ax2.axvline(t_emerge, color='gray', ls='--', lw=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Population')
    ax2.legend(loc='upper right', fontsize=9)
 

    #stats

    final = y[-1]
    peak1 = (y[:, I1]).max(); tpeak1 = t[(y[:, I1]).argmax()]
    peak2 = (y[:, I2]).max(); tpeak2 = t[(y[:, I2]).argmax()]

    stats_lines = [
        f"R0 of variant 1: {R0_1:.2f}",
        f"R0 of variant 2: {R0_2:.2f}",
        f"R0 of variant 2 at emergence: {effective_R0_2:.2f}",
        f"Prevalence Peak V1: {peak1:.1f} at t={tpeak1:.1f}",
        f"Prevalence Peak V2: {peak2:.1f} at t={tpeak2:.1f}\n",
        f"Never infected: {final[S]:.0f}",
    ]

    #checking if the reinfection term was ever meaningfully non zero
    reinfection_flow = beta1 * y[:, S1] * y[:, I1] / N
    reinfection_occured = reinfection_flow.max() > 1e-3

    allowed = omega > 0
    shown = reinfection_occured and allowed
    
    flags_lines = [
        f"Waning immunity: {'Yes' if omega > 0 else 'No'}",
        f"Cross infection: Yes",
        f"Multiple infection allowed in model: {'Yes'}",
        f"Multiple infection shown (same variant): {'Yes' if shown else 'No'}",
    ]

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    still = final[I1] + final[I2]
    if still > 1e-3:
        print(f"(Still infected at end of run: {still:.1f} - should extend t_total)")
    
    ax_text.axis('off')
    ax_text.text(0.5, 0.2, stats_text, fontsize=12, ha='center', va='center',
                transform=ax_text.transAxes,
                bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))

    flags_text = "\n".join(flags_lines)
    ax_flags.axis('off')
    ax_flags.text(0.4, .4, flags_text, fontsize=10, ha='center', va='center',
                  bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))

    # save image
    os.makedirs(figures_dir, exist_ok= True)
    if filename is None:
        filename = f"cross_inf_3_b1_{beta1:.2f}_b2_{beta2:.2f}.png"
    filepath = os.path.join(figures_dir, filename)
    f.savefig(filepath, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    plt.show()
 
 