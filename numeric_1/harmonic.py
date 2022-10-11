"""
This module is a solution to the 'Harmonic oscillator' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1;
    Length in terms of Bohr radii.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.special import eval_hermite

from schrodinger import BOHR_RADII, schrodinger, plot

def V(x: float | np.ndarray) -> float | np.ndarray:
    """
    Potential of oscillator
    """
    return x ** 2 / 2.

def Psi(x: float | np.ndarray, n: int) -> float | np.ndarray:
    """
    Wave function - analytical solution
    TODO - Denne er feil:)
    """
    H_n = eval_hermite(n, x)
    scale_param = np.sqrt(2. ** n * np.math.factorial(n) * np.sqrt(np.pi))
    return H_n * np.exp(-0.5 * x ** 2) / scale_param

def E(n: int) -> float:
    """
    Energy level- analytical solution. In eV.
    """
    return (n + 0.5) * physical_constants['electron volt-hartree relationship'][0]

def main():
    N = 200
    dx = 1e-10 / BOHR_RADII  # 1 Å as bohr radii
    x = np.linspace(-1/2, 1/2, N+1)

    psi_analytical = [Psi(x, n) for n in range(0, 3)]
    analytical_e_levels = [E(n) for n in range(0, 4)]

    energy, psi = schrodinger(V(x/dx), dx)

    plot(
        energy, psi, x, V(x),
        psi_analytical=None, energy_lvls_analytical=analytical_e_levels,
        title='Harmonic oscillator - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
