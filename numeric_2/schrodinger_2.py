"""
This module is an extension of the solution to the first Numerical Assignment.
"""

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def time_evolution(
        energy: np.ndarray, psi: np.ndarray, psi0: np.ndarray
    ) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Parameters
    ----------
    energy : np.ndarray
        Eigenvalues from solving the time independent SE.
    psi : np.ndarray
        Eigenvectors from solving the time independent SE.
    psi0 : np.ndarray
        Wave function at t = 0.

    Returns
    -------
    Psi(x, t) : Callable
        Time evolved function of position x and time t.
    """
    return lambda x, t: np.einsum('ij,i->j', psi, np.dot(psi, psi0) * np.exp(-1j * energy * t))


def animate_wave(
        x: np.ndarray,
        time: float,
        psi: Callable[[np.ndarray, float], np.ndarray],  # x, t -> psi
        fps: int = 25,
        re: bool = False,
        im: bool = False,
        savepath: str = 'animation.gif'
    ) -> None:
    """
    Parameters
    ----------
    x : np.ndarray
        Positional array
    time : float
        Duration of animation in seconds
    psi : Callable[[np.ndarray, float], np.ndarray]
        Time evolved Psi(x, t)
    fps : int
        Frames per second. Default 25
    re, im : bool
        Turn on/off plotting for real and imag. parts of psi. Default false.
    """
    fig, ax = plt.subplots()

    ax.set_xlabel('$x$')
    ax.set_ylabel('$|\Psi|$, $\Re{(\Psi)}$, $\Im{(\Psi)}$')

    ymax = 1.1*np.max(np.abs(psi(x, t=0)))
    graph, = ax.plot([x[0], x[-1]], [0, 2 * ymax])
    if re:
        graph2, = ax.plot([x[0], x[-1]], [0, -2 * ymax])
    if im:
        graph3, = ax.plot([x[0], x[-1]], [0, -2 * ymax])

    time_step = 1. / fps

    def frame(i):
        time = i * time_step
        wave = psi(x, time)
        graph.set_data(x, np.abs(wave))
        graph.set_label(f'$|\Psi(x, t = {time:.2f})|$')
        if re:
            graph2.set_data(x, np.real(wave))
            graph2.set_label(f'$\Re(\Psi(x, t = {time:.2f}))$')
        if im:
            graph3.set_data(x, np.imag(wave))
            graph3.set_label(f'$\Im(\Psi(x, t = {time:.2f}))$')
        ax.legend(loc="upper left")

    FuncAnimation(fig, frame, frames=int(float(time)*fps), interval=time_step*1000, repeat=False).save(
        savepath, writer=PillowWriter(fps=fps)
    )


def main():
    """
    Example given in Jupyter
    """
    animate_wave(
        x = np.linspace(0, 10, 400),
        time = 2.,
        psi = lambda x, t: (x/np.max(x)) * np.exp(1j * x * t),
        re = True,
        im = True,
        savepath='animations/example.gif'
    )

if __name__ == '__main__':
    main()
