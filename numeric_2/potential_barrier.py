"""
This module visualizes the transmittion and reflection of an electron
when colloding with a potential barrier.
"""

import sys
import numpy as np
sys.path.append('..')

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave
from gauss_wave_pkg import psi0

def V(x):
    """
    Potential barrier. V(x) = 1 for x in (0, 1).
    """
    return np.where(np.logical_and(x > 0, x < 1), 10, 0)

def main():
    N = 1000
    x, dx = np.linspace(-10, 10, N, retstep=True)
    energy, psi = schrodinger(V(x), dx)

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x, dx=1., x0=-5, p0=3.), dx),
        savepath='animations/pot_barrier.gif'
    )

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x, dx=1., x0=-5, p0=3.), dx),
        re=True,
        savepath='animations/pot_barrier_2.gif'
    )

if __name__ == '__main__':
    main()
