# Author: Karen Guzman
# Description: Two-variant SIR model with cross-infection, simplified to 5 compartments (S, I1, I2, R1, R2). Recovering from a variant moves you to that variant's R compartment, from which you can still catch the OTHER variant. This model does not track infection history. R1 mixes "variant 1 only" with "variant 2 then variant 1" people, so it cannot report infection-order statistics, and it permits the same variant to be caught twice. Individuals can cycle S->I2->R2->I1->R1->I2->... 
# Use the 9-compartment version if order tracking / strict per-variant immunity is required.

# Date: 6/28/2026
 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
 
# index map
S, I1, I2, R1, R2 = range(5)
 
 
def sim_cross_infection(variables, t, params):
    s, i1, i2, r1, r2 = variables
    beta1, beta2, gamma1, gamma2, eta12, eta21 = params
 
    N = s + i1 + i2 + r1 + r2
 
    # variant 1 spread only by I1, variant 2 spread only by I2
    dSdt  = -beta1 * s * i1 / N - beta2 * s * i2 / N          # S -> I1, S -> I2
    dI1dt =  beta1 * s * i1 / N + eta21 * beta1 * r2 * i1 / N - gamma1 * i1  # (S->I1) + (R2->I1) scaled - recovery
    dI2dt =  beta2 * s * i2 / N + eta12 * beta2 * r1 * i2 / N - gamma2 * i2  # (S->I2) + (R1->I2) scaled - recovery
    dR1dt =  gamma1 * i1 - eta12 * beta2 * r1 * i2 / N               # I1->R1, minus R1->I2 (cross), matches dI2 term
    dR2dt =  gamma2 * i2 - eta21 *beta1 * r2 * i1 / N               # I2->R2, minus R2->I1 (cross), matches dI1 term
 
    return [dSdt, dI1dt, dI2dt, dR1dt, dR2dt]
 
 
def simulator(params, y0, t):
    return integrate.odeint(sim_cross_infection, y0, t, args=(params,))
 
 
def simulate_cross_infection(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
    #Variant 1 circulates alone until t_emerge, then variant 2 is seeded by moving `seed` people from S into I2
    n1 = int(n * t_emerge / t_total) # number of time points before emergence
    n2 = n - n1  # number of time points after emergence
 
    #variant 1 circulates alone until t_emerge, then 'seed' people is from from S into I2
    t1 = np.linspace(0, t_emerge, n1)
    solution1 = simulator(params, y0, t1)
 
    y_handoff = solution1[-1].copy()   # [S, I1, I2, R1, R2] at t_emerge
    y_handoff[S]  -= seed
    y_handoff[I2] += seed
 
    t2 = np.linspace(t_emerge, t_total, n2)
    solution2 = simulator(params, y_handoff, t2)
 
    return np.concatenate([t1, t2]), np.vstack([solution1, solution2])
 
 
def plot_cross_infection(t, y, params, filename = None, figures_dir="figures"):
    S, I1, I2, R1, R2 = range(5) # indices for the compartments
    beta1, beta2, gamma1, gamma2, eta12, eta21 = params
    emerge_index = np.argmax(y[:, I2] > 0) # find the index where the second variant starts to emerge
    t_emerge = t[emerge_index] # get the corresponding time of emergence
    N = y[0].sum()  # total population

    #R0
    R0_1 = beta1 / gamma1
    R0_2 = beta2 / gamma2

    s_emerge = y[emerge_index, S]
    r1_emerge = y[emerge_index, R1]
    effective_R0_2 =  R0_2 * (s_emerge + eta12 * r1_emerge) / N #variant 2 can infect S and R1

    #plot 
    f, (ax_plot, ax_text) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})
    ax_plot.plot(t, y[:, S],  label='Susceptible')
    ax_plot.plot(t, y[:, I1], label='Infected, variant 1')
    ax_plot.plot(t, y[:, R1], label='Recovered, variant 1')
    ax_plot.plot(t[emerge_index:], y[emerge_index:, I2], label='Infected, variant 2')
    ax_plot.plot(t[emerge_index:], y[emerge_index:, R2], label='Recovered, variant 2')
    ax_plot.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of variant 2')
 
    ax_plot.set_xlabel('Time')
    ax_plot.set_ylabel('Population')
    ax_plot.set_title('Emergence of a New Variant with Cross-Infection (5-compartment)')
    ax_plot.legend()

    #stats

    final = y[-1]
    peak1 = (y[:, I1]).max(); tpeak1 = t[(y[:, I1]).argmax()]
    peak2 = (y[:, I2]).max(); tpeak2 = t[(y[:, I2]).argmax()]

    stats_lines = [
        f"R0 of variant 1: {R0_1:.2f}",
        f"R0 of variant 2: {R0_2:.2f}",
        f"R0 of variant 2 at emergence: {effective_R0_2:.2f}",
        f"Peak of variant 1: {peak1:.1f} at t={tpeak1:.1f}",
        f"Peak of variant 2: {peak2:.1f} at t={tpeak2:.1f}\n",
        f"Never infected: {final[S]:.0f}",
        f"Infected by variant 1: {final[R1]:.0f}",
        f"Infected by variant 2: {final[R2]:.0f}",
        f"Total: {final.sum():.0f}",
    ]

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    still = final[I1] + final[I2]
    if still > 1e-3:
        print(f"(Still infected at end of run: {still:.1f} - should extend t_total)")
    
    
    f.subplots_adjust(right=0.75)
    ax_text.axis('off')
    ax_text.text(0.5, -.1, stats_text, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))


    # save image
    os.makedirs(figures_dir, exist_ok= True)
    if filename is None:
        filename = f"cross_inf_2_b1_{beta1:.2f}_b2_{beta2:.2f}.png"
    filepath = os.path.join(figures_dir, filename)
    f.savefig(filepath, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    plt.show()
 
 