import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np
import time

import matplotlib.pyplot as plt

from scipy.optimize import minimize

def vv(x0, v0, force, dt, N, mesh=None, keepvals=None):
    """Velocity verlet integration."""

    x = x0
    v = v0
    a = force(x)
    xx = []
    vv = []
    ee = []
    for i in range(N):
        x += v * dt + 0.5 * a * dt**2
        an = force(x)
        v += 0.5 * (a + an) * dt
        a  = an
        if mesh is om.TriMesh:
            om.write_mesh("out/test"+str(i)+".stl", mesh)
        if hasattr(mesh, '__call__'):
            ee.append(mesh(x,v))
        if keepvals:
            xx.append(x)
            vv.append(v)

    return xx,vv,ee

def test_vv():
    """Verify vv integrator with spring system."""

    k = 1.0
    m = 0.5
    eps = 1.0e-8

    def epot(x):
        return 0.5*k*x**2

    def energy(x,v):
        return epot(x) + 0.5*v**2*m

    def force(x):
        return -(epot(x+eps) - epot(x))/eps
#        return -k/m*x

    x0 = 1.0
    v0 = np.random.normal()

    xn, vn, en = vv(x0,v0, force, 0.01, 1500, mesh=energy, keepvals=True)

    plt.plot(xn,vn)

    plt.figure()
    plt.plot(en)
    plt.title("{}".format(en[-1]))
    plt.show()

def test_energy():
    """Test energy and gradient."""
    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)
    print("Num vertices:",len(points))

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.5, 1.0, 1.0)
    estore.init(mesh)
    print("Time to solution:")

    start = time.time()
    e = m.energy(mesh, estore)
    dt = time.time() - start
    print(" energy functional:", dt)
    print("-")

    gradient1 = np.empty(points.shape)
    gradient2 = np.empty(points.shape)

    # method 1
    start = time.time()
    m.gradient(mesh, estore, gradient1, 1.0e-8)
    dt1 = time.time() - start
    print(" m1 (expected):", dt+len(points)*3*dt)
    print(" m1 (measured):", dt1)
    print("-")

    # method 2
    start = time.time()
    m.s_gradient(mesh, estore, gradient2, 1.0e-6)
    dt2 = time.time() - start
    print(" m2 (expected):", dt+len(points)*(dt/len(points))*3*2*6)
    print(" m2 (measured):", dt2)
    print("-")

    print(np.linalg.norm(gradient1-gradient2))
    plt.plot(gradient1.ravel())
    plt.plot(gradient2.ravel())
    plt.show()

def test_vv_integration():
    """Test md."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.5, 1.0, 1.0)
    estore.init(mesh)

    epot_0 = m.energy(mesh, estore)

    sigma = 0.1

    x = mesh.points()
    v = np.random.normal(size=x.shape)*sigma
    ekin_0 = np.dot(v.ravel(),v.ravel())

    def force(x):
        g = np.empty_like(x)
        m.s_gradient(mesh, estore, g, 1.0e-3)
        return -g*sigma**2

    def energy(q,p):
        return m.energy(mesh, estore) + 0.5*np.dot(p.ravel(), p.ravel())/sigma**2

    res = vv(x, v, force, 0.001, 200, mesh=energy, keepvals=None)

    plt.plot(res[2])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.plot(x[:,0], x[:,1], x[:,2], '.')
    plt.show()

def test_minimization():
    """try direct minimization."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)
    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.5, 1.0, 1.0)
    estore.init(mesh)

    def fun(x):
        points = mesh.points()
        points += x.reshape(points.shape)
        e = m.energy(mesh, estore)
        points -= x.reshape(points.shape)
        return e

    def jac(x):
        points = mesh.points()
        points += x.reshape(points.shape)
        g = np.empty_like(points)
        m.s_gradient(mesh, estore, g, 1.0e-8)
        points -= x.reshape(points.shape)
        return g.ravel()

    x0 = np.zeros_like(points).ravel()
    res = minimize(fun, x0, jac=jac, options={"maxiter": 100})
    print(res.nit, res.message)

    om.write_mesh("out/test0.stl", mesh)
    points = mesh.points()
    points += res.x.reshape(points.shape)
    om.write_mesh("out/test1.stl", mesh)

if __name__ == "__main__":
    #test_energy()
    #test_vv()
    #test_vv_integration()
    test_minimization()
