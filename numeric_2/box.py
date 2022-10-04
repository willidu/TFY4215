"""
part 1
"""
import sys
import numpy as np
sys.path.append('..')  # To enable imports from previous assignment

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave

def main():
    N = 50
    box_length = 5.
    x, dx = np.linspace(0, box_length, N, retstep=True)
    potential = np.zeros_like(x)
    energy, psi = schrodinger(potential, dx)
    psi_t = time_evolution(energy, psi, n=1)

    animate_wave(
        x = x,
        v = potential,
        time = 2.,
        psi = psi_t,
        re = True,
        im = True
    )

if __name__ == '__main__':
    main()
