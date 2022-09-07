"""
This module is a solution to the 'Signle finite well potential' problem.
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

def V(x: float | np.ndarray, v0: float = -10.) -> float | np.ndarray:
    """
    Potential inside well.
    V0 for x in [-L/2, L/2].
    """
    return np.where(np.logical_and(x >= -0.5, x <= 0.5), v0, 0)

def main():
    N = 200
    dx = 1e-12 / physical_constants['Bohr radius'][0]  # 1 picometer as bohr radii
    x = np.linspace(-1, 1, N+1)

    energy, psi = schrodinger(V(x), dx, hartree_atomic_units=True)
    assert np.einsum('ij,ij->i', psi, psi) in np.ones(N+1), 'Wrong scaling of wave functions'

    plot(
        energy, psi, x, V(x), 
        title='Single finite well potential - 1D'
    )

if __name__ == '__main__':
    main()
    plt.show()
