"""
This module is an extension of the solution to the first Numerical Assignment.
"""

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def time_evolution(
        energy: np.ndarray, psi: np.ndarray, n: int = 0
    ) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    Parameters
    ----------
    energy : np.ndarray
        Eigenvalues from solving the time independent SE.
    psi : np.ndarray
        Eigenvectors from solving the time independent SE.
    n : int
        Quantum number +1 (counting from 0).

    Returns
    -------
    Psi_n(x, t) : Callable
        Time evolved function of position x and time t.
    """
    return lambda x, t: np.einsum('ij,i->j', psi, np.dot(psi, psi[n]) * np.exp(-1j * energy * t))


def animate_wave(
        x: np.ndarray,
        v: np.ndarray,
        time: float,
        psi: Callable[[np.ndarray, float], np.ndarray],  # x, t -> psi
        fps: int = 25,
        re: bool = False,
        im: bool = False,
        savepath: str = 'animation.gif'
    ):
    """
    Parameters
    ----------
    x : np.ndarray
        Positional array
    v : np.ndarray
        Potential in position x
    time : float
        Duration of animation in seconds
    psi : Callable[[np.ndarray, float], np.ndarray]
        Time evolved Psi(x, t)
    fps : int
        Frames per second. Default 25
    re, im : bool
        Turn on/off plotting for real and imag. parts of psi. Default false.
    """
    time_step = 1. / fps
    fig, ax = plt.subplots()

    ax.set_xlabel("$x$")
    ax.set_ylabel("$|\Psi|$, $\Re{(\Psi)}$, $\Im{(\Psi)}$")

    ymax = max(np.max(np.abs(psi(x, 0))**2), np.max(np.abs(psi(x, 0))))
    graph, = ax.plot([x[0], x[-1]], [0, +2*ymax])
    if re:
        graph2, = ax.plot([x[0], x[-1]], [0, -2*ymax])
    if im:
        graph3, = ax.plot([x[0], x[-1]], [0, -2*ymax])

    ax2 = ax.twinx()
    v_max = np.min(v) + 1.1 * (np.max(v) - np.min(v)) + 1 # + 1 if v = const
    x_ext = np.concatenate(([x[0]], x, [x[-1]]))
    v_ext = np.concatenate(([v_max], v, [v_max]))
    ax2.set_ylabel("$V(x)$")
    ax2.plot(x_ext, v_ext, linewidth=3, color="black", label="V")
    ax2.legend(loc="upper right")

    def frame(i):
        time = i * time_step
        wave = psi(x, time)
        graph.set_data(x, np.abs(wave))
        graph.set_label(f"$|\Psi(x, t = {time:.2f})|$")
        if re:
            graph2.set_data(x, np.real(wave))
            graph2.set_label(f"$\Re(\Psi(x, t = {time:.2f}))$")
        if im:
            graph3.set_data(x, np.imag(wave))
            graph3.set_label(f"$\Im(\Psi(x, t = {time:.2f}))$")
        ax.legend(loc="upper left")

    FuncAnimation(fig, frame, frames=int(time*fps), interval=time_step*1000, repeat=False).save(
        savepath, writer=PillowWriter(fps=fps)
    )


def main():
    """
    Example given in Jupyter
    """
    animate_wave(
        x = np.linspace(0, 10, 400),
        v = np.zeros(400),
        time = 2.,
        psi = lambda x, t: (x/np.max(x)) * np.exp(1j * x * t),
        re = True,
        im = True,
        savepath='animations/example.gif'
    )


if __name__ == '__main__':
    main()
