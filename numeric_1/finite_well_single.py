"""
This module is a solution to the 'Signle finite well potential' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1;
    Length in terms of Bohr radii.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import BOHR_RADII, schrodinger, plot

def V(x: float | np.ndarray, v0: float = -10.) -> float | np.ndarray:
    """
    Potential inside well.
    V0 for x in [-L/2, L/2].
    """
    return np.where(np.logical_and(x >= -0.5, x <= 0.5), v0, 0)

def main():
    N = 200
    dx = 1e-12 / BOHR_RADII  # 1 picometer as bohr radii
    x = np.linspace(-1, 1, N+1)

    energy, psi = schrodinger(V(x), dx)

    plot(
        energy, psi, x, V(x),
        title='Single finite well potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
