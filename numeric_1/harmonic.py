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
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
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

    # Short test to check scaling of wave func. / prob. since it does not
    # match the analytical solution
    assert np.einsum('ij,ij->i', psi, psi) in np.ones(N+1), 'Wrong scaling of wave functions'

    plot(
        energy, psi, x, V(x), 
        psi_analytical=None, energy_lvls_analytical=analytical_e_levels,
        title='Harmonic oscillator - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
