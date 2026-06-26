# Author: Karen Guzman
# Description: Simulating the SIR model of an infectious disease with the possibility of cross-infection. Full cross-immunity is not assumed: recovering from the first variant does not grant immunity to the second variant, so individuals can be infected by the second variant after recovering from the first.
# Date: 6/26/2026

# For the first version of the cross-infection model, we do not assume waning immunity. So an individual is immune to the first variant after recovering from it, but can still be infected by the second variant. In a future version, we will add waning immunity to the model.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

#redefine our sim function
def sim_cross_infection(variables, t, params):
    S, I1, I2, R1, R2 = variables
    beta1, beta2, gamma1, gamma2 = params

    N = S + I1 + I2 + R1 + R2
    #adding cross-infection dynamics, after recovering from the first variant, individuals can be infected by the second variant
    dSdt = -beta1 * S * I1 / N - beta2 * S * I2 / N
    dI1dt = beta1 * S * I1 / N + beta1 * R2 * I1 / N - gamma1 * I1 #group is made up of individuals susceptible to the first variant and those who have recovered from the second variant
    dI2dt = beta2 * S * I2 / N + beta2 * R1 * I2 / N - gamma2 * I2  #group is made up of individuals susceptible to the second variant and those who have recovered from the first variant
    dR1dt = gamma1 * I1 - beta2 * R1 * I2 / N  #group is made up of individuals who have recovered from the first variant minus those who have been infected by the second variant
    dR2dt = gamma2 * I2 - beta1 * R2 * I1 / N  #group is made up of individuals who have recovered from the second variant minus those who have been infected by the first variant

    return ([dSdt, dI1dt, dI2dt, dR1dt, dR2dt])

#generic integration function
def simulator(params, y0, t):
    return integrate.odeint(sim_cross_infection, y0, t, args=(params,))
    #returns the number of people in each compartment at each time point in t

#emergence: variant 1 alone until t_emerge, then variant 2 emerges
def simulate_cross_infection(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
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

    return np.concatenate([t1, t2]), np.vstack([solution1, solution2]) # t is the time vector, y is the solution of the ODEs for all time points in the form: [S, I1, I2, R1, R2]

#plotting function
def plot_cross_infection(t, y):
    emerge_index = np.argmax(y[:,2] > 0)  # find the index where the second variant starts to emerge
    t_emerge = t[emerge_index]  # get the corresponding time of emergence
    
    #plot
    plt.figure(figsize=(10,6))
    plt.plot(t, y[:,0], label='Susceptible')
    plt.plot(t, y[:,1], label='Infected with Strain 1')
    plt.plot(t, y[:,3], label='Recovered from Strain 1')
    plt.plot(t[emerge_index:], y[emerge_index:,2], label='Infected with Strain 2')
    plt.plot(t[emerge_index:], y[emerge_index:,4], label='Recovered from Strain 2')

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of Strain 2')

    plt.title('Emergence of a New Strain in an Epidemic')
    plt.legend()
    
    #print statistics about the epidemic
    print(f"Peak of Strain 1: {np.max(y[:,1])} at time {t[np.argmax(y[:,1])]}")
    print(f"Peak of Strain 2: {np.max(y[:,2])} at time {t[np.argmax(y[:,2])]}")
    #print numbers for all scenarios: 
    # 1. individuals do not become infected by either variant
    # 2. individuals become infected by the first variant but not the second
    # 3. individuals become infected by the second variant but not the first
    # 3. individuals are first infected with variant 1, then 2
    # 4. individuals are first infected with variant 2, then 1
    print(f"Individuals who do not become infected by either variant: {y[-1,0]}")
    print(f"Individuals who become infected by the first variant but not the second: {y[-1,3]}")
    print(f"Individuals who become infected by the second variant but not the first: {y[-1,4]}")
    print(f"Individuals who are first infected with variant 1, then 2: {y[-1,4] - y[-1,3]}")
    print(f"Individuals who are first infected with variant 2, then 1: {y[-1,3] - y[-1,4]}")
    plt.show()
