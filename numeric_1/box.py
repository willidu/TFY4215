"""
This module is a solution to the traditional 'Particle in a box' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import schrodinger, plot

def V(x):
    """
    Potential inside the box
    """
    return np.zeros_like(x)


def Psi(x, n, L=1.):
    """
    Wave function - analytical solution
    """
    return np.sqrt(2./L) * np.sin(n*x*np.pi/L)


def E(n, L=1.):
    """
    Energy level- analytical solution in Hatree.
    """
    return n ** 2 * np.pi ** 2 / (2. * L ** 2)


def main():
    N = 100
    x, dx = np.linspace(0, 1, N, retstep=True)
    energy, psi = schrodinger(V(x), dx)

    plot(
        energy, psi, x, V(x),
        psi_analytical = [Psi(x, n) for n in range(1, 4)],
        energy_lvls_analytical = E(np.arange(1, 5)),
        title = 'One dimentional infinite well'
    )


if __name__ == '__main__':
    main()
    plt.show()
