# Author: Karen Guzman
# Description: Simulating the SIR model of an infectious disease with the new emergence of a variant. The parameter beta2 is the transmission of this new variant, which is assumed to be more transmissible than the first variant (beta2 > beta1). Full cross-immunity is assumed: recovering from either variant grants immunity to BOTH, so each individual is infected at most once.
# Date: 6/19/2026

# Difference from cross-infection (next model):
# Here, we assume that having either variant makes you immune to both. The two variants compete for the same susceptible population. In the cross-infection model, individuals can be infected by the second variant after recovering from the first.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

#redefine our sim function
def sim_new_emergence(variables, t, params):
    S, I1, I2, R1, R2 = variables
    beta1, beta2, gamma1, gamma2 = params

    N = S + I1 + I2 + R1 + R2
    # new emergence of a variant 
    dSdt = -beta1 * S * I1 / N - beta2 * S * I2 / N
    dI1dt = beta1 * S * I1 / N - gamma1 * I1
    dI2dt = beta2 * S * I2 / N - gamma2 * I2
    dR1dt = gamma1 * I1
    dR2dt = gamma2 * I2

    return ([dSdt, dI1dt, dI2dt, dR1dt, dR2dt])

#generic integration function
def simulator(params, y0, t):
    return integrate.odeint(sim_new_emergence, y0, t, args=(params,))
    #returns the number of people in each compartment at each time point in t

#emergence: variant 1 alone until t_emerge, then variant 2 emerges
def simulate_emergence(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
    # returns (t, y) where t is the time vector and y is the solution of the ODEs
    n1 = int(n * t_emerge / t_total)  # number of time points before emergence
    n2 = n - n1  # number of time points after emergence

    t1 = np.linspace(0, t_emerge, n1)
    solution1 = simulator(params, y0, t1)
    y_handoff = solution1[-1].copy()  #grabs last row of solution1, which is list of [S, I1, I2, R1, R2] at t_emerge
    y_handoff[0] -= seed  # reduce susceptible population by seed amount
    y_handoff[2] += seed  # increase infected population of variant 2 by seed amount

    t2 = np.linspace(t_emerge, t_total, n2)
    solution2 = simulator(params, y_handoff, t2)

    return np.concatenate([t1, t2]), np.vstack([solution1, solution2]) # t is the time vector, y is the solution of the ODEs for all time points


#plotting function
def plot_emergence(t, y, params, filename = None, figures_dir="figures"):
    beta1, beta2, gamma1, gamma2 = params

    emerge_index = np.argmax(y[:,2] > 0)  # find the index where the second variant starts to emerge
    t_emerge = t[emerge_index]  # get the corresponding time of emergence

    #calculate R0 for both variants and effective R0 for variant 2 at the time of emergence
    R0_1 = beta1 / gamma1
    R0_2 = beta2 / gamma2 #beta2 /gamma2 is variant 2's R0 in a fully susceptible population, but the effective R0 will be lower because some people are already immune to both variants
    s_emerge = y[emerge_index,0]  # susceptible population at the time of emergence
    N = y[0].sum()  # total population (conserved)
    effective_R0_2 = R0_2 * (s_emerge / N)  # effective R0 of variant 2 at the start of its emergence; S/N
    
    #plot
    # figure with 2 rows: plot on top, stats strip below
    f, (ax_plot, ax_text) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={'height_ratios': [4, 1]}
    )
    ax_plot.plot(t, y[:,0], label='Susceptible')
    ax_plot.plot(t, y[:,1], label='Infected with Strain 1')
    ax_plot.plot(t, y[:,3], label='Recovered from Strain 1')
    ax_plot.plot(t[emerge_index:], y[emerge_index:,2], label='Infected with Strain 2')
    ax_plot.plot(t[emerge_index:], y[emerge_index:,4], label='Recovered from Strain 2')

    ax_plot.set_xlabel('Time')
    ax_plot.set_ylabel('Population')
    ax_plot.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of Strain 2')

    ax_plot.set_title('Emergence of a New Strain in an Epidemic')
    ax_plot.legend()

    stats_lines = [
        f"R0 (variant 1): {R0_1:.2f}",
        f"R0 (variant 2): {R0_2:.2f}",
        f"Effective R0 (variant 2 at emergence): {effective_R0_2:.2f}",
        f"Peak of Strain 1: {y[:,1].max():.0f} at t={t[y[:,1].argmax()]:.1f}",
        f"Peak of Strain 2: {y[:,2].max():.0f} at t={t[y[:,2].argmax()]:.1f}",
        f"Total infected with Strain 1: {y[-1,3]:.0f}",
        f"Total infected with Strain 2: {y[-1,4]:.0f}",
        f"Never infected (final S): {y[-1,0]:.0f}"
    ]
    
    stats_text = "\n".join(stats_lines)
    print(stats_text)

    
    
    f.subplots_adjust(right=0.75)
    ax_text.axis('off')
    ax_text.text(0.5, .3, stats_text, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))

    # save image
    os.makedirs(figures_dir, exist_ok= True)
    if filename is None:
        filename = f"new_emergence_b2_{beta2:.3f}.png"
    filepath = os.path.join(figures_dir, filename)
    f.savefig(filepath, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    plt.show()