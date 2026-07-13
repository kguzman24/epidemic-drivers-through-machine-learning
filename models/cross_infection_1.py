# Author: Karen Guzman
# Description: Simulating the SIR model of an infectious disease with cross-infection and potential cross-immunity. Recovering from one variant grants immunity to that variant and reduced susceptibility to the other, controlled by cross-immunity factors eta12 and eta21.  η₁₂, η₂₁ ∈ [0,1] and  η = 1 means there is no cross-protection. η = 0 means full cross protection. This version does not include WANING IMMUNITY.
# Date: 6/26/2026

# For the first version of the cross-infection model, we do not assume waning immunity. So an individual is immune to the first variant after recovering from it, but can still be infected by the second variant. In a future version, we will add waning immunity to the model.


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os

#redefine our sim function
def sim_cross_infection(variables, t, params):
    S, I1, I2, R1, R2, I12, I21, R12, R21 = variables
    #the variables will have final states that mimic 5 scenarios: S = never infected, I1 = variant 1 only, I2 = variant 2 only, R1 = recovered from variant 1 only, R2 = recovered from variant 2 only, I12 = had variant 1-then-2, I21 = had variant 2-then-1, R12 = recovered from variant 1-then-2, R21 = recovered from variant 2-then-1
    beta1, beta2, gamma1, gamma2, eta12, eta21 = params

    N = S + I1 + I2 + R1 + R2 + I12 + I21 + R12 + R21  #total population

    #adding cross-infection dynamics, after recovering from the first variant, individuals can be infected by the second variant
    dSdt = -beta1 * S * (I1 + I21) / N - beta2 * S * (I2 + I12) / N
    dI1dt = beta1 * S * (I1 + I21 )/ N - gamma1 * I1 #infected with variant 1 only
    dI2dt = beta2 * S * (I2 + I12) / N - gamma2 * I2  #infected with variant 2 only
    dR1dt = gamma1 * I1 - eta12 * beta2 * R1 * (I2 + I12) / N  #recovered from variant 1 only minus those who have been infected by the second variant
    dR2dt = gamma2 * I2 - eta21 * beta1 * R2 * (I1 + I21) / N  #recovered from variant 2 only minus those who have been infected by the first variant
    dI12dt = eta12 * beta2 * R1 * (I2 + I12) / N - gamma2 * I12  #R2 people catching variant 1
    dI21dt = eta21 * beta1 * R2 * (I1 + I21) / N - gamma1 * I21 #R1 people catching variant 2
    dR12dt = gamma2 * I12 #recovered from both, variant 1 first
    dR21dt = gamma1 * I21 #recovered from both, variant 2 first

    return ([dSdt, dI1dt, dI2dt, dR1dt, dR2dt, dI12dt, dI21dt, dR12dt, dR21dt])

#generic integration function
def simulator(params, y0, t):
    return integrate.odeint(sim_cross_infection, y0, t, args=(params,))
    #returns the number of people in each compartment at each time point in t

#emergence: variant 1 alone until t_emerge, then variant 2 emerges
def simulate_cross_infection(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
    # returns (t, y) where t is the time vector and y is the solution of the ODEs

    n1 = int(n * t_emerge / t_total)  # number of time points before emergence
    n2 = n - n1  # number of time points after emergence

    #variant 1 circulates alone until t_emerge, then 'seed' people is from from S into I2
    t1 = np.linspace(0, t_emerge, n1)
    solution1 = simulator(params, y0, t1)
    y_handoff = solution1[-1].copy()  #grabs last row of solution1, which is list of [S, I1, I2, R1, R2, I12, I21, R12, R21] at t_emerge
    y_handoff[0] -= seed  # reduce susceptible population by seed amount
    y_handoff[2] += seed  # increase infected population of variant 2 by seed amount

    t2 = np.linspace(t_emerge, t_total, n2)
    solution2 = simulator(params, y_handoff, t2)

    return np.concatenate([t1, t2]), np.vstack([solution1, solution2]) # t is the time vector, y is the solution of the ODEs for all time points in the form: [S, I1, I2, R1, R2, I12, I21, R12, R21]

#plotting function
def plot_cross_infection(t, y, params, filename = None, figures_dir="figures"):
    S, I1, I2, R1, R2, I12, I21, R12, R21 = range(9)  # indices for the compartments
    beta1, beta2, gamma1, gamma2, eta12, eta21 = params
    emerge_index = np.argmax(y[:,2] > 0)  # find the index where the second variant starts to emerge
    t_emerge = t[emerge_index]  # get the corresponding time of emergence

    # R0
    R0_1 = beta1 / gamma1
    R0_2 = beta2 / gamma2 #beta2 /gamma2 is variant 2's R0 in a fully susceptible population, but the effective R0 will be different because variant 2 can infect S and R1
    N = y[0].sum()  # total population (conserved)
    s_emerge = y[emerge_index,S]  # susceptible population at the time of emergence
    r1_emerge = y[emerge_index,R1]  #recovered from variant 1 only at the time of emergence
    effective_R0_2 = R0_2 * (s_emerge + r1_emerge) / N  # effective R0 of variant 2 at the start of its emergence; S/N + R1/N

    #plot
    f, (ax_plot, ax_text) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={'height_ratios': [4, 1]}
    )
    ax_plot.plot(t, y[:,S], label='Susceptible') #[:, S]
    ax_plot.plot(t, y[:,I1], label='Infected, variant 1 only') #[:, I1]
    ax_plot.plot(t, y[:,R1], label='Recovered, variant 1 only') #[:, R1]
    ax_plot.plot(t[emerge_index:], y[emerge_index:,I2], label='Infected, variant 2 only') #[:, I2]
    ax_plot.plot(t[emerge_index:], y[emerge_index:,R2], label='Recovered, variant 2 only') #[:, R2]
    ax_plot.plot(t[emerge_index:], y[emerge_index:,I12], label='Infected, variant 1 then 2') #[:, I12]
    ax_plot.plot(t[emerge_index:], y[emerge_index:,I21], label='Infected, variant 2 then 1') #[:, I21]
    ax_plot.plot(t[emerge_index:], y[emerge_index:,R12] + y[emerge_index:, R21], label = 'Recovered from both', linestyle='--') #[:, R12 + R21]
   
    ax_plot.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of Variant 2')

    ax_plot.set_xlabel('Time')
    ax_plot.set_ylabel('Population')


    ax_plot.set_title('Emergence of a New Variant with Cross-Infection')
    ax_plot.legend(loc='upper right')

    final = y[-1]
    peak1 = (y[:, I1] + y[:, I21]).max(); tpeak1 = t[(y[:, I1] + y[:, I21]).argmax()]
    peak2 = (y[:, I2] + y[:, I12]).max(); tpeak2 = t[(y[:, I2] + y[:, I12]).argmax()]

    stats_lines = [
        f"R0 of variant 1: {R0_1:.2f}",
        f"R0 of variant 2: {R0_2:.2f}",
        f"R0 of variant 2 at emergence: {effective_R0_2:.2f}",
        f"Peak of variant 1 (I1+I21): {peak1:.1f} at t={tpeak1:.1f}",
        f"Peak of variant 2 (I2+I12): {peak2:.1f} at t={tpeak2:.1f}\n",
        f"Never infected: {final[S]:.0f}",
        f"Infected by variant 1 only: {final[R1]:.0f}",
        f"Infected by variant 2 only: {final[R2]:.0f}",
        f"First variant 1, then variant 2: {final[R12]:.0f}",
        f"First variant 2, then variant 1: {final[R21]:.0f}",
        f"Total: {final.sum():.0f}",
    ]

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    still = final[I1] + final[I2] + final[I12] + final[I21]
    if still > 1e-3:
        print(f"(Still infected at end of run: {still:.1f} - should extend t_total)")
    
    
    f.subplots_adjust(right=0.75)
    ax_text.axis('off')
    ax_text.text(0.5, 0, stats_text, fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))

    # save image
    os.makedirs(figures_dir, exist_ok= True)
    if filename is None:
        filename = f"cross_inf1_b1_{beta1:.2f}_b2_{beta2:.2f}.png"
    filepath = os.path.join(figures_dir, filename)
    f.savefig(filepath, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    plt.show()