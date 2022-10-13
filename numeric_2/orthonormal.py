"""
This module is for testing the orthonorma properties of stationary solutions to
the Schrödinger equation. Normality is checked in the solver from numeric_1,
so we only need to test the orthogonality of solutions.
"""

import sys
import numpy as np
sys.path.append('..')

from numeric_1.schrodinger import schrodinger


def main():
    x, dx = np.linspace(-20, +20, 400, retstep=True)

    potentials = [
        np.zeros_like(x),                                            # Infinite well
        x ** 2 / 4,                                                  # Harmonic oscillator
        np.piecewise(x, [np.abs(x) > 2, np.abs(x) <= 2], [0, -10]),  # Finite well
        np.random.rand(len(x))                                       # Random
    ]

    for psi in [schrodinger(v, dx)[1] for v in potentials]:

        assert np.allclose(
            np.transpose(psi) @ psi,
            np.identity(psi.shape[0])/dx
        ), 'Psi not orthogonal'

if __name__ == '__main__':
    main()
