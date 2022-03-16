import helfrich as m
import meshzoo
import meshio
import numpy as np
import time

from util import get_energy_manager

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import truncnorm

def vv(x0, v0, force, m, dt, N):
    """Velocity verlet integration."""

    x = x0
    v = v0
    a = force(x,v)
    xx = [x.copy()]
    vv = [v.copy()]
    for i in range(N):
        x += (v * dt + 0.5 * a * dt**2) / m
        an = force(x,v)
        v += 0.5 * (a + an) * dt
        a  = an
        xx.append(x.copy())
        vv.append(v.copy())

    return np.array(xx),np.array(vv)

def hybrid_move(x, force, m, dt, L, T=1):
    """Hybrid move."""
    v = np.random.normal(size=x.shape)*np.sqrt(m*T)
    tforce = lambda q,p: force(q,p,T)
    xn, vn = vv(x, v, tforce, m, dt, L)
    return xn[-1], v, vn[-1]

def hmc(x0, nlog_hamil, force, m, N, dt, L, cT, info=None, istep=10):
    """Hybrid monte carlo."""

    x = x0.copy()
    xx = []
    acc = 0
    T = 1.0
    for i in range(N):

        # move around
        xn, v, vn = hybrid_move(x, force, m, dt, L, T)

        de = nlog_hamil(xn,vn,T) - nlog_hamil(x,v,T)

        # compute acceptance probability
        a = min(1, np.exp(-de))
        u = np.random.uniform()
        if u<=a:
            x = xn
            xx.append(xn.copy())
            acc += 1

        if i>0 and (i%istep) == 0 and not info is None:
            info(i,xn,vn,T,acc/(i+1))

        T = 1.0 * np.exp(-cT*i)

    rate = acc/N
    print("acc-rate: {}".format(rate))

    return np.array(xx)

def test_vv():
    """Verify vv integrator with spring system."""

    k = 1.0
    m = 0.5
    eps = 1.0e-8

    def energy(x,v):
        return 0.5*k*x**2 + 0.5*v**2/m

    def force(x,v):
#        return -k*x
        return -(energy(x+eps,v) - energy(x,v))/eps

    x0 = np.array([1.0])
    v0 = np.array([1.0])

    xn, vn = vv(x0, v0, force, m, 0.1, 100)
    en = [energy(xi,vi) for xi,vi in zip(xn,vn)]

    plt.plot(xn,vn)
    plt.figure()
    plt.plot(en)
    plt.title("{}".format(en[-1]))
    plt.show()

def test_integration():
    """Test md."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = m.TriMesh(points, cells)

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                1.0, 1.0e4, 1.0e4, 0.0, 10,
                                1.0, 0.8, 1.0)
    estore.print_info(mesh)

    sigma = 1 #-> m
    gamma = 1.0

    x = mesh.points()
    v = np.random.normal(size=x.shape)*np.sqrt(sigma)

    def force(x,v):
        g = estore.gradient(mesh)
        return -g-gamma*v

    xn, vn = vv(x, v, force, sigma, 0.001, 20000)

    n = 0
    for i,xi in enumerate(xn):
        if i%200 == 0:
            np.copyto(x,xi)
            meshio.write_points_cells("out/test_"+str(n)+".stl",
                                      mesh.points(),
                                      [('triangle', mesh.fv_indices())])
            n += 1

    estore.print_info(mesh)

def test_hmc():
    """Test hamilton monte carlo."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = m.TriMesh(points, cells)

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                1.0, 1.0e4, 1.0e4, 0.0, 10,
                                1.0, 0.8, 1.0)
    estore.print_info(mesh)

    sigma = 1 #-> m

    x0 = mesh.points().copy()

    def hamiltonian(x,v,T):
        points = mesh.points()
        np.copyto(points, x)
        vr = v.ravel()
        e = estore.energy(mesh)/T + 0.5*vr.dot(vr)/sigma/T
        np.copyto(points,x0)
        return e

    def force(x,v,T):
        points = mesh.points()
        np.copyto(points,x)
        g = estore.gradient(mesh)
        np.copyto(points,x0)
        return -g/T

    def print_info(i,x,v,T,acc):
        print("\n-- Step ",i)
        print("  ----- Temperature:", T)
        print("  ----- acc-rate:   ", acc)
        p = mesh.points()
        np.copyto(p,x)
        estore.energy(mesh)
        estore.print_info(mesh)

    def flip():
        m.flip_edges(mesh)

    # break somewhere > 7000
    xn = hmc(x0, hamiltonian, force, sigma, 10000, 0.001, 10, 0.001, info=print_info, istep=100)

    for i,xi in enumerate(xn):
        if i%100 == 0:
            x = mesh.points()
            np.copyto(x,xi)
            meshio.write_points_cells("out/test_"+str(i)+".stl",
                                      mesh.points(),
                                      [('triangle', mesh.fv_indices())])

def test_minimization():
    """try direct minimization."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = m.TriMesh(points, cells)

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                1.0, 1.0e4, 1.0e4, 0.0, 10,
                                1.0, 0.8, 1.0)
    estore.print_info(mesh)

    def fun(x):
        points = mesh.points()
        points += x.reshape(points.shape)
        e = estore.energy(mesh)
        points -= x.reshape(points.shape)
        return e

    def jac(x):
        points = mesh.points()
        points += x.reshape(points.shape)
        g = estore.gradient(mesh)
        points -= x.reshape(points.shape)
        return g.ravel()

    x0 = np.zeros_like(points).ravel()
    res = minimize(fun, x0, jac=jac, options={"maxiter": 5000, "disp": 1})
    print(res.nit, res.message)

    meshio.write_points_cells("out/test0.stl",
                              mesh.points(),
                              [('triangle', mesh.fv_indices())])
    points = mesh.points()
    points += res.x.reshape(points.shape)
    meshio.write_points_cells("out/test1.stl",
                              mesh.points(),
                              [('triangle', mesh.fv_indices())])

    estore.print_info(mesh)

if __name__ == "__main__":
    #test_vv()
    #test_integration()
    #test_minimization()
    test_hmc()
