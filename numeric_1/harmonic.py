"""
This module is a solution to the 'Harmonic oscillator' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1;
    Omega = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite

from schrodinger import schrodinger, plot

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
    Energy level- analytical solution in Hartree.
    """
    return (n + 0.5)


def main():
    N = 1000
    x, dx = np.linspace(-3, 3, N, retstep=True)
    energy, psi = schrodinger(V(x), dx)

    plot(
        energy, psi, x, V(x),
        psi_analytical=[Psi(x, n) for n in range(0, 4)],
        energy_lvls_analytical=[E(n) for n in range(0, 4)],
        title='Harmonic oscillator - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
