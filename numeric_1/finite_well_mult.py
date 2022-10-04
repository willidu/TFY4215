"""
This module is a solution to the 'Multiple finite wells potential' problem.
Uses Hartree atomic units:
    Electron mass = 1;
    Elementary charge = 1;
    Reduced Plack's constant = 1;
    Length in terms of Bohr radii.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger import BOHR_RADII, schrodinger, plot

def V(x: float | np.ndarray, v0: float = -100., cutoff: float = 5.) -> float | np.ndarray:
    """
    Potential inside wells.
    """
    w = 1.  # Box width
    g = .8  # Separation distance
    inside_box = np.where(x/(w+g) - np.floor(x/(w+g)) < w/(w+g), v0, 0)
    return np.where(np.logical_and(x > -cutoff, x < cutoff), inside_box, 0.)

def main():
    N = 1000
    dx = .1e-12 / BOHR_RADII  # .1 picometer as bohr radii
    x = np.linspace(-1, 1, N+1)
    box_scale_factor = 10  # Yields [-10, 10] * dx -> box 2 picometers wide

    energy, psi = schrodinger(V(x * box_scale_factor), dx)

    plot(
        energy, psi, x, V(x * box_scale_factor), 
        title='Multiple finite wells potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
