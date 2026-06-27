# Author: Karen Guzman
# Description: Simulating the SIR model of an infectious disease with the possibility of cross-infection. Full cross-immunity is not assumed: recovering from the first variant does not grant immunity to the second variant, so individuals can be infected by the second variant after recovering from the first.
# Date: 6/26/2026

# For the first version of the cross-infection model, we do not assume waning immunity. So an individual is immune to the first variant after recovering from it, but can still be infected by the second variant. In a future version, we will add waning immunity to the model.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#redefine our sim function
def sim_cross_infection(variables, t, params):
    S, I1, I2, R1, R2, I12, I21, R12, R21 = variables
    #the variables will have final states that mimic 5 scenarios: S = never infected, I1 = variant 1 only, I2 = variant 2 only, R1 = recovered from variant 1 only, R2 = recovered from variant 2 only, I12 = had variant 1-then-2, I21 = had variant 2-then-1, R12 = recovered from variant 1-then-2, R21 = recovered from variant 2-then-1
    beta1, beta2, gamma1, gamma2 = params

    N = S + I1 + I2 + R1 + R2 + I12 + I21 + R12 + R21  #total population

    #adding cross-infection dynamics, after recovering from the first variant, individuals can be infected by the second variant
    dSdt = -beta1 * S * (I1 + I21) / N - beta2 * S * (I2 + I12) / N
    dI1dt = beta1 * S * (I1 + I21 )/ N - gamma1 * I1 #infected with variant 1 only
    dI2dt = beta2 * S * (I2 + I12) / N - gamma2 * I2  #infected with variant 2 only
    dR1dt = gamma1 * I1 - beta2 * R1 * (I2 + I12) / N  #recovered from variant 1 only minus those who have been infected by the second variant
    dR2dt = gamma2 * I2 - beta1 * R2 * (I1 + I21) / N  #recovered from variant 2 only minus those who have been infected by the first variant
    dI12dt = beta2 * R1 * (I2 + I12) / N - gamma2 * I12  #R2 people catching variant 1
    dI21dt = beta1 * R2 * (I1 + I21) / N - gamma1 * I21 #R1 people catching variant 2
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
def plot_cross_infection(t, y):
    S, I1, I2, R1, R2, I12, I21, R12, R21 = range(9)  # indices for the compartments
    emerge_index = np.argmax(y[:,2] > 0)  # find the index where the second variant starts to emerge
    t_emerge = t[emerge_index]  # get the corresponding time of emergence
    
    #plot
    plt.figure(figsize=(10,6))
    plt.plot(t, y[:,S], label='Susceptible') #[:, S]
    plt.plot(t, y[:,I1], label='Infected, variant 1 only') #[:, I1]
    plt.plot(t, y[:,R1], label='Recovered, variant 1 only') #[:, R1]
    plt.plot(t[emerge_index:], y[emerge_index:,I2], label='Infected, variant 2 only') #[:, I2]
    plt.plot(t[emerge_index:], y[emerge_index:,R2], label='Recovered, variant 2 only') #[:, R2]
    plt.plot(t[emerge_index:], y[emerge_index:,I12], label='Infected, variant 1 then 2') #[:, I12]
    plt.plot(t[emerge_index:], y[emerge_index:,I21], label='Infected, variant 2 then 1') #[:, I21]
    plt.plot(t[emerge_index:], y[emerge_index:,R12] + y[emerge_index:, R21], label = 'Recovered from both', linestyle='--') #[:, R12 + R21]
   
    plt.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of Variant 2')

    plt.xlabel('Time')
    plt.ylabel('Population')


    plt.title('Emergence of a New Variant with Cross-Infection')
    plt.legend()
    
    #print statistics about the epidemic
    print(f"Peak of variant 1 (I1+I21): {np.max(y[:, I1] + y[:, I21]):.1f} "
          f"at time {t[np.argmax(y[:, I1] + y[:, I21])]:.1f}")
    print(f"Peak of variant 2 (I2+I12): {np.max(y[:, I2] + y[:, I12]):.1f} "
          f"at time {t[np.argmax(y[:, I2] + y[:, I12])]:.1f}")
    #print numbers for all scenarios: 
    # 1. individuals do not become infected by either variant
    # 2. individuals become infected by the first variant but not the second
    # 3. individuals become infected by the second variant but not the first
    # 3. individuals are first infected with variant 1, then 2
    # 4. individuals are first infected with variant 2, then 1
    final = y[-1]
    print("\nScenarios at the end of the simulation: ")
    print(f"1. Never infected by either variant: {final[S]:.1f}")
    print(f"2. Infected by variant 1 only:        {final[R1]:.1f}")
    print(f"3. Infected by variant 2 only:        {final[R2]:.1f}")
    print(f"4. First variant 1, then variant 2:   {final[R12]:.1f}")
    print(f"5. First variant 2, then variant 1:   {final[R21]:.1f}")
    still = final[I1] + final[I2] + final[I12] + final[I21]
    if still > 1e-3:
        print(f"   (still infected at end of run:    {still:.1f} - extend t_total)")
    print(f"   total accounted for: {final.sum():.1f}")
 
    plt.show()
