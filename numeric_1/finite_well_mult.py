"""
This module is a solution to the 'Multiple finite wells potential' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import schrodinger, plot

def V(
        x: float | np.ndarray, v0: float = -100., cutoff: float = 5.,
        width: float = 1., sep: float = 1.
    ) -> float | np.ndarray:
    """
    Potential inside wells.
    """
    inside_box = np.where(x/(width+sep) - np.floor(x/(width+sep)) < width/(width+sep), v0, 0)
    return np.where(np.logical_and(x > 0., x < cutoff), inside_box, 0.)


def main():
    N = 1000
    x, dx = np.linspace(-2, 7, N, retstep=True)
    potential = V(x, v0=-40)
    energy, psi = schrodinger(potential, dx)

    plot(
        energy, psi, x, potential,
        title='Multiple finite wells potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
