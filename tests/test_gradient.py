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
    a = force(x,v)
    xx = []
    vv = []
    ee = []
    for i in range(N):
        x += v * dt + 0.5 * a * dt**2
        an = force(x,v)
        v += 0.5 * (a + an) * dt
        a  = an
        if isinstance(mesh, om.TriMesh): # and i%20 == 0:
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
    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])

    params = m.BondParams()
    params.b = 1.0
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 1.33*a
    params.lmin = 0.67*a

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.5, 1.0, 1.0, params)
    estore.init(mesh)
    print("Time to solution:")

    start = time.time()
    e = m.energy(mesh, estore)
    dt = time.time() - start
    print(" energy functional:", dt)
    print("-")

    gradient = np.empty(points.shape)

    # method 2
    start = time.time()
    m.gradient(mesh, estore, gradient, 1.0e-6)
    dt2 = time.time() - start
    print(" m2 (expected):", dt+len(points)*(dt/len(points))*3*2*6)
    print(" m2 (measured):", dt2)
    print("-")

    plt.plot(gradient.ravel())
    plt.show()

def plot_tether():
    """Plot tether potential."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    print(a)
    params = m.BondParams()
    params.b = 1.0
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 1.33*a
    params.lmin = 0.67*a
    print(params.lc0, params.lmax)
    print(params.lc1, params.lmin)

    # repelling part
    xa = np.linspace(params.lmin-0.003, params.lc1, 1000)
    repel = np.zeros_like(xa)
    repel = params.b * np.exp(0.01/(xa-params.lc1))/(xa-params.lmin)

    # attractive
    xb = np.linspace(params.lc0, params.lmax+0.004, 1000)
    attr = np.zeros_like(xb)
    attr = params.b * np.exp(0.01/(params.lc0-xb))/(params.lmax-xb)

    plt.plot(xa,repel)
    plt.plot(xb, attr)
    plt.axvline(params.lmax, color="g", lw=0.3, label="lmax")
    plt.axvline(params.lmin, color="g", lw=0.3, label="lmin")
    plt.axvline(params.lc0, color="r", lw=0.3, label="lc0")
    plt.axvline(params.lc1, color="r", lw=0.3, label="lc1")
    plt.axvline(a, color="k", lw=0.3, label="a")
    plt.title("Noguchi tether potential")
    plt.legend()
    plt.show()

def test_integration():
    """Test md."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.type = "tether"
    params.b = 1.0
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 1.33*a
    params.lmin = 0.67*a

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.8, 1.0, 1.0, params)
    estore.init(mesh, ref_lambda=1.0)

    epot_0 = m.energy(mesh, estore)
    estore.print_info("")

    sigma = 0.5
    gamma = 0.0

    x = mesh.points()
    v = np.random.normal(size=x.shape)*sigma
    ekin_0 = np.dot(v.ravel(),v.ravel())

    def force(x,v):
        g = np.empty_like(x)
        m.gradient(mesh, estore, g, 1.0e-6)
        return -g-gamma*v

    res = vv(x, v, force, 0.001, 4000, mesh, keepvals=None)

    estore.print_info("")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.plot(x[:,0], x[:,1], x[:,2], '.')
    plt.show()


def test_minimization():
    """try direct minimization."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    params = m.BondParams()
    params.b = 1.0e2
    estore = m.EnergyValueStore(1.0, 1.0e2, 1.0e4, 0.0, 0.5, 1.0, 1.0, params)
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
        m.gradient(mesh, estore, g, 1.0e-6)
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
    #plot_tether()
    test_integration()
    #test_minimization()
