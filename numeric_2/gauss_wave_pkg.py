"""
This module visualizes an electron in an infinite well, representing a free electron by
a Gaussian wave packet.
"""

import sys
import numpy as np
sys.path.append('..')

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave


def psi0(x: np.ndarray, dx: float, x0: float = 0., p0: float = 0.) -> np.ndarray:
    """
    Gaussian wave packet Psi(x, t=0).

    Parameters
    ----------
    x : np.ndarray
        Position
    dx : float
        Uncertainty in x direction
    x0 : float
        Initial position. Default 0.
    p0 : float
        Initial momentum. Default 0.
    """
    return np.exp(-(x - x0) ** 2 / (4 * dx ** 2)) \
        * np.exp(1j * p0 * x) \
        / (2 * np.pi * dx ** 2) ** (1 / 4)


def main():
    N = 1000
    # x, dx = np.linspace(0, 25, N, retstep=True)
    x, dx = np.linspace(0, 15, N, retstep=True)
    potential = np.zeros_like(x)
    energy, psi = schrodinger(potential, dx)

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x, dx=1., x0=5, p0=3.), dx),
        savepath='animations/wavepkg.gif'
    )

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x, dx=1., x0=5, p0=3.), dx),
        re = True,
        savepath='animations/wavepkg_2.gif'
    )

if __name__ == '__main__':
    main()
