"""
This module is an implementation for solving the time-independent Schrödinger equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

COLORS = ['g', 'b', 'r', 'm']

def schrodinger(
        potential: np.ndarray,
        dx: float
    ) -> tuple[np.ndarray]:
    """
    Solves the time independent Schrödinger equation with potential V(x).
    Returns energy in eV. Assumes constant distance between points.

    Parameters
    ----------
    potential : np.ndarray
        V(x) evaluated at positions.
    dx : float
        Distance between in terms of Bohr radii.

    Returns
    -------
    E : np.ndarray
        Energy levels in Hartree (eigenvalues)
    Psi : np.ndarray
        Normalized wave functions (eigenfunctions)
    """
    diag = 1. / (dx ** 2) + potential
    semidiag = - 1. / (2. * dx ** 2) * np.ones(len(potential) - 1)
    eigvals, eigvecs = eigh_tridiagonal(diag, semidiag)
    eigvecs = eigvecs.T / np.sqrt(dx)
    assert np.allclose(np.einsum('ij,ij->i', eigvecs, eigvecs)*dx, np.ones(len(eigvecs))), 'Psi not normalized'
    return eigvals, eigvecs


def plot(
        energy: np.ndarray,
        psi: np.ndarray,
        x: np.ndarray,
        potential: np.ndarray,
        psi_analytical: None | list[float] = None,
        energy_lvls_analytical: None | list[float] = None,
        title: None | str = None
    ) -> None:
    """
    TODO
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=30, fontweight='semibold')
    fig.subplots_adjust(hspace=.4, wspace=0.3)

    # Upper left plot
    for i in range(len(COLORS)):
        ax[0,0].plot(x, psi[i], label=f'n = {i+1}', color=COLORS[i])

    if psi_analytical is not None:
        for n in range(min(len(psi_analytical), len(COLORS))):
            ax[0,0].plot(x, psi[n], label=f'$\psi_{n+1}$', color=COLORS[n], ls='--')

    ax[0,0].set(
        ylabel=r'$\Psi$',
        xlabel='Distance x/L',
        xlim=(np.min(x), np.max(x)),
        title='Wave function'
    )
    ax[0,0].legend(ncol=4, loc='lower left')
    ax[0,0].grid(True)

    # Upper right plot
    for i in range(min(len(COLORS), 3)):
        ax[0,1].plot(x, psi[i]**2, label=f'n = {i+1}', color=COLORS[i])

    ax[0,1].set(
        ylabel=r'${|\Psi|}^2$',
        xlabel='Distance x/L',
        xlim=(np.min(x), np.max(x)),
        title='Probability density'
    )
    ax[0,1].legend(loc='lower center', ncol=3)
    ax[0,1].grid(True)

    # Lower left plot
    for i in range(len(COLORS)):
        ax[1,0].axhline(energy[i], label=f'n = {i+1}', color=COLORS[i])

    # Plots analytical levels if given
    if energy_lvls_analytical is not None:
        for n in range(min(len(energy_lvls_analytical), len(COLORS))):
            ax[1,0].axhline(energy_lvls_analytical[n], label=f'$E_{n+1}$', color=COLORS[n], ls='--')

    ax[1,0].set(
        xlabel='Distance x/L',
        ylabel='Energy [Hartree]',
        title='Energy levels'
    )
    ax[1,0].legend(loc='upper left', ncol=4)
    ax[1,0].grid(True)

    # Lower right plot
    ax[1,1].plot(x, potential, color='g')
    ax[1,1].set(
        xlabel='Distance x',
        ylabel='V(x)',
        xlim=(np.min(x), np.max(x)),
        title='Potential V(x)'
    )
    ax[1,1].grid(True)
