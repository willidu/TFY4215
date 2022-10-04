"""
This module is for testing the orthonorma properties of stationary solutions to
the Schrödinger equation.
"""
import sys
import numpy as np
sys.path.append('..') # To enable imports from previous assignment

from numeric_1.schrodinger import schrodinger

def main():
    x, dx = np.linspace(-20, +20, 400, retstep=True)

    potentials = [
        0. * x,                                                      # Infinite well
        1. / 4.*x**2,                                                # Harmonic oscillator
        np.piecewise(x, [np.abs(x) > 2, np.abs(x) <= 2], [0, -10]),  # Finite well
        np.random.rand(len(x))                                       # Random
    ]

    for psi in [schrodinger(v, dx)[1] for v in potentials]:
        assert np.allclose(np.linalg.inv(psi), np.transpose(psi), rtol=1e-5)            # Orthogonality
        assert np.allclose(np.einsum('ij,ij->i', psi,psi), np.ones_like(x), rtol=1e-5)  # Norm

if __name__ == '__main__':
    main()
