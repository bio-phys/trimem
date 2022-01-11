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
                                0.1, 1.0e2, 1.0e2, 1.0e2, 1.0e3, 0.1, # weights
                                0.8, .8, 0.8)                 # fractions

    estore.print_info()

    print("Time to solution:")

    start = time.time()
    e = estore.energy()
    dt = time.time() - start
    print(" energy           :", dt)
    print("-")

    gradient1 = np.empty(points.shape)

    # method 1 (fd)
    start = time.time()
    m.gradient(mesh, estore, gradient1, 1.0e-8)
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
    plt.savefig("res.pdf")
    print(np.linalg.norm(gradient1-gradient2)/np.linalg.norm(gradient1))
    plt.show()

if __name__ == "__main__":
    test_energy()
