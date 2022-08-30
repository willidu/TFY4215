import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants

def planck(wavelengths, temperature):
    return (2 * np.pi * scipy.constants.h * scipy.constants.c ** 2 / wavelengths ** 5) \
        / (np.exp(scipy.constants.c * scipy.constants.h / (scipy.constants.Boltzmann * temperature * wavelengths)) - 1.)

def main():
    wavelengths = np.linspace(0, 5e-6, 10_00)

    for temp in (3000, 4000, 5000):
        res = planck(wavelengths, temp)
        plt.plot(wavelengths/1e-6, res, label=f'{temp:.0f} K')
        plt.axhline(np.max(res), ls='--', color='k', alpha=0.5, linewidth=1)

    plt.legend(loc='upper right')
    plt.xlim((0, wavelengths.max()/1e-6))
    plt.xlabel(r'Wavelength [$\mu m$]')
    plt.ylabel(r'Spectral radiance [W m$^{-2}$ m]')
    plt.grid(True)

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
    plt.show()
