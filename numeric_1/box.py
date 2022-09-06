"""
This module is a solution to the traditional 'Particle in a box' problem.
Uses Hartree atomic units.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

from schrodinger import schrodinger, plot

def V(x):
    """
    Potential inside the box
    """
    return np.zeros_like(x)

def Psi(x, n, L):
    """
    Wave function - analytical solution
    """
    return np.sqrt(2./L) * np.sin(n*x*np.pi/L)

def E(n, L):
    """
    Energy level- analytical solution. In eV.
    """
    return n ** 2 * np.pi ** 2 / (2. * L ** 2 * physical_constants['electron volt-hartree relationship'][0])

def main():
    N = 100
    box_length = 4e-9 / physical_constants['Bohr radius'][0]  # 4 nm as bohr radii
    x = np.linspace(0, box_length, N+1)  # Burde gå [0, 1] heller

    psi_analytical = np.asarray([Psi(x, n, L=box_length) for n in range(1, 4)])
    analytical_e_levels = E(np.arange(1, 5), L=box_length)

    energy, psi = schrodinger(V(x), box_length/(N+1), True)
    plot(energy, psi, x/box_length, V(x), psi_analytical, analytical_e_levels)

if __name__ == '__main__':
    main()
    plt.show()
