import helfrich as m
import meshzoo
import numpy as np

import matplotlib.pyplot as plt

def plot_tether():
    """Plot tether potential."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = m.TriMesh(points, cells)

    a,_ = m.avg_tri_props(mesh)
    b = 1.0
    lc0 = 1.15*a
    lc1 = 0.85*a
    lmax = 1.33*a
    lmin = 0.67*a

    # repelling part
    eps = 1.0e-3
    xa = np.linspace(lmin-eps, lc1-eps**2, 1000)
    repel = np.zeros_like(xa)
    repel = b * np.exp(1/(xa-lc1))/(xa-lmin)

    # attractive
    xb = np.linspace(lc0+eps**2, lmax+eps, 1000)
    attr = np.zeros_like(xb)
    attr = b * np.exp(1/(lc0-xb))/(lmax-xb)

    plt.plot(xa,repel)
    plt.plot(xb, attr)
    plt.axvline(lmax, color="g", lw=0.3, label="lmax")
    plt.axvline(lmin, color="g", lw=0.3, label="lmin")
    plt.axvline(lc0, color="r", lw=0.3, label="lc0")
    plt.axvline(lc1, color="r", lw=0.3, label="lc1")
    plt.axvline(a, color="k", lw=0.3, label="a")
    plt.title("Noguchi tether potential")
    plt.legend()
    plt.show()

def plot_mod_tether():
    """Plot tether potential."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = m.TriMesh(points, cells)

    a,_ = m.avg_tri_props(mesh)
    b = 100.0
    r = 2
    lc0 = 1.15*a
    lc1 = 0.85*a

    # repelling part
    xa = np.linspace(0+0.0001, lc1-0.0001, 1000)
    repel = np.zeros_like(xa)
    repel = b * np.exp(xa/(xa-lc1))*xa**(-r)

    # attractive
    xb = np.linspace(lc0+0.0001, 4*a, 1000)
    attr = np.zeros_like(xb)
    attr = b * r**(r+1)*(xb-lc0)**(r)

    plt.plot(xa,repel)
    plt.plot(xb, attr)
    plt.axvline(0.0, color="r", lw=0.3, label="min")
    plt.axvline(lc0, lw=0.3, label="lc0")
    plt.axvline(lc1, lw=0.3, label="lc1")
    plt.axvline(a, color="k", lw=0.3, label="a")
    plt.title("Modified Noguchi tether potential")
    plt.ylim([-10, 500])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_tether()
    plot_mod_tether()
