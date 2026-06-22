# Author: Karen Guzman
# Description: Simulating the SIR model of infectious disease with waning immunity. The waning immunity parameter is the rate at which recovered individuals lose immunity and become susceptible again.
# Date: 6/18/2026
# The basic SIR model implementation was adapted from the YouTube tutorial "SIR Epidemiology Model with Python" by Mike Saint-Antoine (2021), available at https://www.youtube.com/watch?v=cbXCyO_F2v8.

from scipy import integrate
import matplotlib.pyplot as plt

# define function
def sim(variables, t, params):
    S, I, R = variables
    beta, gamma, omega = params

    N = S + I + R
    dSdt = -beta * S * I / N + omega * R
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I - omega * R

    return ([dSdt, dIdt, dRdt])

# solve the system of equations
def simulator(params, y0, t):
    return integrate.odeint(sim, y0, t, args=(params,))

# plotting function
def plot_simulation(t, y, params):
    # plot the data
    beta, gamma, omega = params
    R0 = beta / gamma

    #define figure with 3 subplots
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))


    line1 = ax1.plot(t, y[:, 0], 'b', label='Susceptible') #0 column with all rows
    line2 = ax2.plot(t, y[:, 1], 'r', label='Infected')
    line3 = ax3.plot(t, y[:, 2], 'g', label='Recovered')

    ax1.set_title('SIR Model with Waning Immunity')
    ax1.set_xlabel('Time')
    ax3.set_ylabel('Number of Individuals')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    #add stats
    N = y[0].sum()   # total population (conserved)

    print(f"Peak number of infected: {y[:,1].max():.0f}")
    print(f"Time to peak: {t[y[:,1].argmax()]:.1f}")
    print(f"Basic reproduction number (R0): {R0:.2f}")
    if omega > 0:
        print(f"Endemic equilibrium (S): {N/R0:.1f}")
        print(f"Endemic equilibrium (I): {(N - N/R0)/(1 + gamma/omega):.1f}")
    else:
        print("No waning (omega = 0): disease burns out.")
        print(f"Final susceptibles remaining: {y[-1,0]:.0f}")
        print(f"Final infected: {y[-1,1]:.1f}")
    plt.show()