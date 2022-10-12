"""
This module is a solution to the 'Signle finite well potential' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import schrodinger, plot

def V(x: float | np.ndarray, v0: float = -10.) -> float | np.ndarray:
    """
    Potential inside well.
    V0 for x in [-1/2, 1/2].
    """
    return np.where(np.logical_and(x >= -0.5, x <= 0.5), v0, 0)


def main():
    N = 1000
    x, dx = np.linspace(-1, 1, N, retstep=True)
    potential = V(x, v0=-40)  # V0 = -40 yields three bound states
    energy, psi = schrodinger(potential, dx)

    plot(
        energy, psi, x, potential,
        title='Single finite well potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
