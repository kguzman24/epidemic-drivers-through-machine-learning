# Author: Karen Guzman
# Description: Simulating the SIR model of infectious disease with waning immunity. The waning immunity parameter is the rate at which recovered individuals lose immunity and become susceptible again.
# Date: 6/18/2026
# Credit: https://www.youtube.com/watch?v=cbXCyO_F2v8 

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
def plot_simulation(t, y, beta, gamma):
    # plot the data
    R0 = beta / gamma

    #define figure with 3 subplots
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))


    line1 = ax1.plot(t, y[:, 0], 'b', label='Susceptible') #0 column with all rows
    line2 = ax2.plot(t, y[:, 1], 'r', label='Infected')
    line3 = ax3.plot(t, y[:, 2], 'g', label='Recovered')

    ax1.set_title('SIR Model with Waning Immunity')
    ax1.set_xlabel('Time')
    ax3.set_ylabel('Number of Individuals')
    plt.legend()
    print(f'R0: {R0:.2f}')
    plt.show()