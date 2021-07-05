import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np
import time

from util import get_energy_manager

import matplotlib.pyplot as plt

def test_energy():
    """Test energy and gradient."""
    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)
    print("Num vertices:",len(points))

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                1.0, 1.0e4, 1.0e4, 1.0, 1.0, # weights
                                0.5, 1.0, 1.0)               # fractions

    print("Time to solution:")

    start = time.time()
    e = estore.energy()
    dt = time.time() - start
    print(" energy           :", dt)
    print("-")

    gradient1 = np.empty(points.shape)

    # method 1 (using locality of energy functional but not parallelized)
    start = time.time()
    m.gradient(mesh, estore, gradient1, 1.0e-6)
    dt2 = time.time() - start
    print(" gradient (fd)    :", dt2)
    print("-")

    # method 2 (exact)
    start = time.time()
    gradient2 = estore.gradient()
    dt2 = time.time() - start
    print(" gradient (exact) :", dt2)
    print("-")

    plt.plot(gradient1.ravel())
    plt.plot(gradient2.ravel())
    print(np.linalg.norm(gradient1-gradient2))
    plt.show()

if __name__ == "__main__":
    test_energy()
