"""
part 1
"""
import sys
import numpy as np
sys.path.append('..')  # To enable imports from previous assignment

from numeric_1.schrodinger import schrodinger
from schrodinger_2 import time_evolution, animate_wave

def main():
    N = 100
    box_length = 11.
    x, dx = np.linspace(0, box_length, N, retstep=True)
    potential = np.zeros_like(x)
    energy, psi = schrodinger(potential, dx)

    time_period = 2 * np.pi / (energy[1] - energy[0])
    print(f'T = {time_period:.2f}')

    animate_wave(
        x = x,
        v = potential,
        time = time_period,
        psi = lambda x, t: (time_evolution(energy, psi, psi0=psi[0])(x, t) \
            + time_evolution(energy, psi, psi0=psi[1])(x, t)) / np.sqrt(2),
        savepath='animations/box_superpos.gif'
    )

    animate_wave(
        x = x,
        v = potential,
        time = 3*time_period,
        psi = lambda x, t: (time_evolution(energy, psi, psi0=psi[0])(x, t) \
            + time_evolution(energy, psi, psi0=psi[1])(x, t)) / np.sqrt(2),
        re = True,
        im = True,
        savepath='animations/box_superpos_2.gif'
    )

if __name__ == '__main__':
    main()
