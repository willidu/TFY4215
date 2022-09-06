"""
This module is an implementation for solving the time-independent Schrödinger equation.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from scipy.linalg import eigh_tridiagonal

COLORS = ['g', 'b', 'r', 'c']

def schrodinger(
        potential: np.ndarray,
        dx: float,
        hartree_atomic_units: bool = True
    ) -> tuple[np.ndarray]:
    """
    Solves the time independent Schrödinger equation with potential V(x).
    Returns energy in eV regardless of atomic or SI units.
    Assumes constant distance between points.

    Parameters
    ----------
    potential : np.ndarray
        V(x) evaluated at positions.
    dx : float
        Distance between points either in meters or bohr radii.
    hartree_atomic_units : bool, optional
        Switches from SI units to Hartree atomic units. Default True.

    Returns
    -------
    E : np.ndarray
        Energy levels in eV (eigenvalues)
    Psi : np.ndarray
        Wave functions (eigenfunctions)
    """
    if hartree_atomic_units:
        hbar = 1.
        mass = 1.
        e_scale = scipy.constants.physical_constants['electron volt-hartree relationship'][0]
    else:
        hbar = scipy.constants.hbar
        mass = scipy.constants.electron_mass
        e_scale = scipy.constants.elementary_charge

    diag = hbar ** 2 / (mass * dx ** 2) + potential
    semidiag = - hbar ** 2 / (2. * mass * dx ** 2) * np.ones(len(potential) - 1)
    eigvals, eigvecs_norm = eigh_tridiagonal(diag, semidiag)
    return eigvals / e_scale, eigvecs_norm.T


def plot(
        energy: np.ndarray,
        psi: np.ndarray,
        x: np.ndarray,
        potential: np.ndarray,
        psi_analytical: None | list[float] = None,
        energy_lvls_analytical: None | list[float] = None
    ) -> None:
    """
    TODO
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Particle in a box - 1D', fontsize=30, fontweight='semibold')
    fig.subplots_adjust(hspace=.4, wspace=0.3)

    # Upper left plot
    for i in range(3):
        ax[0,0].plot(x, psi[i], label=f'n = {i+1}', color=COLORS[i])

    if psi_analytical is not None:
        for n, psi_n in enumerate(psi_analytical):
            ax[0,0].plot(x, psi_n, label=f'$\psi_{n+1}$', color=COLORS[n], ls='--')

    ax[0,0].set(
        ylabel=r'$\Psi$',
        xlabel='Distance x/L',
        xlim=(np.min(x), np.max(x)),
        title='Wave function'
    )
    ax[0,0].legend(ncol=3)
    ax[0,0].grid(True)

    # Upper right plot
    for i in range(3):
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
    for i in range(4):
        ax[1,0].axhline(energy[i], label=f'n = {i+1}', color=COLORS[i])

    # Plots analytical levels if given
    if energy_lvls_analytical is not None:
        for i, energy_level in enumerate(energy_lvls_analytical):
            ax[1,0].axhline(energy_level, label=f'$E_{i+1}$', color=COLORS[i], ls='--')

    ax[1,0].set(
        xlabel='Distance x/L',
        ylabel='Energy [eV]',
        xlim=(np.min(x), np.max(x)),
        ylim=(0, 1.55*energy[i]),  # Makes room for legend at the top
        title='Energy levels'
    )
    ax[1,0].legend(loc='upper left', ncol=4)
    ax[1,0].grid(True)

    # Lower right plot
    ax[1,1].plot(x, potential, color='g')
    ax[1,1].set(
        xlabel='Distance x/L',
        ylabel='V(x)',
        xlim=(np.min(x), np.max(x)),
        title='Potential energy'
    )
    ax[1,1].grid(True)
