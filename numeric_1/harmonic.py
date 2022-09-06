"""
This module is a solution to the 'Harmonic oscillator' problem.
Uses Hartree atomic units:
    omega = 1;
    mass = 1;
    elementary charge = 1;
    reduced placks constant = 1;
    length in terms of bohr radii.
"""

import numpy as np
from numpy.polynomial.hermite import hermval
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

from schrodinger import schrodinger, plot

def hermite(x: float, n: int) -> float:
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return hermval(x, herm_coeffs)

def V(x: float | np.ndarray) -> float | np.ndarray:
    """
    Potential inside the box
    """
    return x ** 2 / 2.


def Psi(x: float | np.ndarray, n: int) -> float | np.ndarray:
    """
    Wave function - analytical solution
    TODO - Denne er feil:)
    """
    return hermite(x, n) * np.exp(- x ** 2 / 2.) / (np.pi ** 0.25 * np.sqrt(np.math.factorial(n) * 2. ** n))

def E(n: int) -> float:
    """
    Energy level- analytical solution. In eV.
    """
    return (n + 0.5) * physical_constants['electron volt-hartree relationship'][0]

def main():
    N = 200
    dx = 1e-10 / physical_constants['Bohr radius'][0]  # 1 Å as bohr radii
    x = np.linspace(-1/2, 1/2, N+1)

    psi_analytical = [Psi(x, n) for n in range(0, 3)]
    analytical_e_levels = [E(n) for n in range(0, 4)]

    energy, psi = schrodinger(V(x/dx), dx, hartree_atomic_units=True)
    plot(
        energy, psi, x, V(x), 
        psi_analytical=None, energy_lvls_analytical=analytical_e_levels
    )

if __name__ == '__main__':
    main()
    plt.show()
