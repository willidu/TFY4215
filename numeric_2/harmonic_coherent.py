"""
This module visualizes an electron in an harmonic oscillator.
Uses m = omega = hbar = 1.
"""

import sys
import numpy as np
sys.path.append('..')

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave

def V(x: float | np.ndarray, L: float = 1.) -> float | np.ndarray:
    """
    Potential of oscillator. Centered around L/2.
    """
    return (x - L/2) ** 2 / 2.


def psi0(x: np.ndarray, x0: float = 0.) -> np.ndarray:
    """
    Translated stationary state wave Psi(x, t=0).
    """
    return np.power(np.pi, -1/4) * np.exp((-(x - x0) ** 2) / 2.)


def main():
    N = 1000
    box_length = 15.
    x, dx = np.linspace(0, box_length, N, retstep=True)
    energy, psi = schrodinger(V(x, L=box_length), dx)

    animate_wave(
        x = x,
        time = 10.,
        psi = time_evolution(energy, psi, psi0(x, x0=0.75*box_length), dx),
        savepath='animations/harm.gif'
    )

if __name__ == '__main__':
    main()
