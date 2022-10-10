"""
part 2
TODO
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('..')  # To enable imports from previous assignment

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave


def V(x: float | np.ndarray, v0: float = -10., L: float = 1.) -> float | np.ndarray:
    """
    Potential inside well.
    V0 for x in [-L/2, L/2].
    """
    return np.where(np.logical_and(x >= -L/2, x <= L/2), v0, 0)


def main():
    N = 100
    box_length = 100.
    x, dx = np.linspace(-box_length/2, box_length/2, N, retstep=True)
    potential = np.zeros_like(x)
    energy, psi = schrodinger(potential, dx)

    def psi0(x: np.ndarray) -> np.ndarray:
        """
        Gaussian wave packet Psi(x, t=0).
        """
        hbar = 1
        x0 = 0.
        p0 = 0.
        a = 1. / (2 * np.pi * dx ** 2) ** (1. / 4.)
        b = np.exp(-(x-x0)**2 / (4 * dx ** 2))
        c = np.exp(-1j * p0 * x / hbar)
        return a * b * c

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x)),
        savepath='animations/wavepkg.gif'
    )

    animate_wave(
        x = x,
        time = 5.,
        psi = time_evolution(energy, psi, psi0(x)),
        re = True,
        im = True,
        savepath='animations/wavepkg_2.gif'
    )

if __name__ == '__main__':
    main()
