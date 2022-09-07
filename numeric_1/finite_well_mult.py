"""
This module is a solution to the 'Multiple finite wells potential' problem.
Uses Hartree atomic units:
    mass = 1;
    elementary charge = 1;
    reduced placks constant = 1;
    length in terms of bohr radii.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants

from schrodinger import schrodinger, plot

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
    dx = .1e-12 / physical_constants['Bohr radius'][0]  # .1 picometer as bohr radii
    x = np.linspace(-1, 1, N+1)
    box_scale_factor = 10  # Yields [-10, 10] * dx -> box 2 picometers wide

    energy, psi = schrodinger(V(x * box_scale_factor), dx, hartree_atomic_units=True)
    assert np.einsum('ij,ij->i', psi, psi) in np.ones(N+1), 'Wrong scaling of wave functions'

    plot(
        energy, psi, x, V(x * box_scale_factor), 
        title='Multiple finite wells potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
