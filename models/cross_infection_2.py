# Author: Karen Guzman
# Description: Two-variant SIR model with cross-infection, simplified to 5 compartments (S, I1, I2, R1, R2). Recovering from a variant moves you to that variant's R compartment, from which you can still catch the OTHER variant. This model does not track infection history. R1 mixes "variant 1 only" with "variant 2 then variant 1" people, so it cannot report infection-order statistics, and it permits the same variant to be caught twice. Individuals can cycle S->I2->R2->I1->R1->I2->... 
# Use the 9-compartment version if order tracking / strict per-variant immunity is required.
# Date: 6/28/2026
 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
 
# index map
S, I1, I2, R1, R2 = range(5)
 
 
def sim_cross_infection(variables, t, params):
    s, i1, i2, r1, r2 = variables
    beta1, beta2, gamma1, gamma2 = params
 
    N = s + i1 + i2 + r1 + r2
 
    # variant 1 spread only by I1, variant 2 spread only by I2
    dSdt  = -beta1 * s * i1 / N - beta2 * s * i2 / N          # S -> I1, S -> I2
    dI1dt =  beta1 * s * i1 / N + beta1 * r2 * i1 / N - gamma1 * i1  # (S->I1) + (R2->I1) - recovery
    dI2dt =  beta2 * s * i2 / N + beta2 * r1 * i2 / N - gamma2 * i2  # (S->I2) + (R1->I2) - recovery
    dR1dt =  gamma1 * i1 - beta2 * r1 * i2 / N               # I1->R1, minus R1->I2 (cross)
    dR2dt =  gamma2 * i2 - beta1 * r2 * i1 / N               # I2->R2, minus R2->I1 (cross)
 
    return [dSdt, dI1dt, dI2dt, dR1dt, dR2dt]
 
 
def simulator(params, y0, t):
    return integrate.odeint(sim_cross_infection, y0, t, args=(params,))
 
 
def simulate_cross_infection(params, y0, t_emerge, t_total=300, n=1000, seed=1.0):
    #Variant 1 circulates alone until t_emerge, then variant 2 is seeded by moving `seed` people from S into I2
    n1 = int(n * t_emerge / t_total)
    n2 = n - n1
 
    t1 = np.linspace(0, t_emerge, n1)
    solution1 = simulator(params, y0, t1)
 
    y_handoff = solution1[-1].copy()   # [S, I1, I2, R1, R2] at t_emerge
    y_handoff[S]  -= seed
    y_handoff[I2] += seed
 
    t2 = np.linspace(t_emerge, t_total, n2)
    solution2 = simulator(params, y_handoff, t2)
 
    return np.concatenate([t1, t2]), np.vstack([solution1, solution2])
 
 
def plot_cross_infection(t, y):
    emerge_index = np.argmax(y[:, I2] > 0)
    t_emerge = t[emerge_index]
 
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, S],  label='Susceptible')
    plt.plot(t, y[:, I1], label='Infected, variant 1')
    plt.plot(t, y[:, R1], label='Recovered, variant 1')
    plt.plot(t[emerge_index:], y[emerge_index:, I2], label='Infected, variant 2')
    plt.plot(t[emerge_index:], y[emerge_index:, R2], label='Recovered, variant 2')
    plt.axvline(x=t_emerge, color='r', linestyle='--', label='Emergence of variant 2')
 
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Emergence of a New Variant with Cross-Infection (5-compartment)')
    plt.legend()
 
    print(f"Peak of variant 1: {np.max(y[:, I1]):.1f} at time {t[np.argmax(y[:, I1])]:.1f}")
    print(f"Peak of variant 2: {np.max(y[:, I2]):.1f} at time {t[np.argmax(y[:, I2])]:.1f}")
    print(f"Never infected by either variant (final S): {y[-1, S]:.1f}")
    print(f"Total population (conservation check): {y[-1].sum():.1f}")
 
    plt.show()
 
 