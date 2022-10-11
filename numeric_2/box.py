"""
This module shows the time evolution of a particle in a box with a superposition
of the ground state and first excited state.
"""

import sys
import numpy as np
sys.path.append('..')  # To enable imports from previous assignment

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave


def psi0(psi):
    """
    Superposition of Psi0(x) and Psi1(x)
    """
    return (psi[0] + psi[1]) / np.sqrt(2)


def main():
    N = 100
    box_length = 11.
    x, dx = np.linspace(0, box_length, N, retstep=True)
    potential = np.zeros_like(x)
    energy, psi = schrodinger(potential, dx)

    time_period = 2 * np.pi / (energy[1] - energy[0])  # [s]
    print(f'T = {time_period:.2f} s')

    animate_wave(
        x = x,
        time = time_period,
        psi = time_evolution(energy, psi, psi0(psi)),
        savepath='animations/box_superpos.gif'
    )

    animate_wave(
        x = x,
        time = 3*time_period,
        psi = time_evolution(energy, psi, psi0(psi)),
        re = True,
        im = True,
        savepath='animations/box_superpos_2.gif'
    )

if __name__ == '__main__':
    main()
