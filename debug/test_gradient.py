import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np
import time

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
    v = np.random.normal(size=x.shape)*np.sqrt(m)
    tforce = lambda q,p: force(q,p,T)
    xn, vn = vv(x, v, tforce, m, dt, L)
    return xn[-1], v, vn[-1]

def hmc(x0, nlog_hamil, force, m, N, dt, L, info=None, istep=10):
    """Hybrid monte carlo."""

    x = x0.copy()
    xx = []
    acc = 0
    T = 1.0
    dT = T/N
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

        T -= dT

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

def test_neighbourhood():
    """test neighbourhood extraction."""

    points, cells = meshzoo.icosa_sphere(1)
    mesh = om.TriMesh(points, cells)

    om.write_mesh("out/mesh.stl", mesh)

    vh = mesh.vertex_handle(1)
    patch = m.get_neighbourhood_copy(mesh, vh)
    om.write_mesh("out/patch.stl", patch)

    print("n_faces    :", patch.n_faces())
    print("n_halfedges:",len([he for he in patch.halfedges()]))

    # check copying
    ppoints = patch.points()
    print(points)
    print(ppoints)

    ppoints *= 0
    print(points)
    print(ppoints)

def test_energy():
    """Test energy and gradient."""
    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)
    print("Num vertices:",len(points))
    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])

    params = m.BondParams()
    params.type = "tether"
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

    gradient1 = np.empty(points.shape)
    gradient2 = np.empty(points.shape)
    gradient3 = np.empty(points.shape)

    # method 1 (using locality of energy functional but not parallelized)
    start = time.time()
    m.gradient(mesh, estore, gradient1, 1.0e-6)
    dt2 = time.time() - start
    print(" m1 (expected):", dt+len(points)*(dt/len(points))*3*2*6)
    print(" m1 (measured):", dt2)
    print("-")

    # method 2 (using plain energy functional)
    start = time.time()
    m.s_gradient(mesh, estore, gradient2, 1.0e-6)
    dt2 = time.time() - start
    print(" m2 (expected):", dt+len(points)*3*dt)
    print(" m2 (measured):", dt2)
    print("-")

    # method 3 (using locality of the energy functional + parallelization)
    start = time.time()
    m.f_gradient(mesh, estore, gradient3, 1.0e-6)
    dt2 = time.time() - start
    print(" m3 (expected):", dt+len(points)*(dt/len(points))*3*2*6)
    print(" m3 (measured):", dt2)
    print("-")

    plt.plot(gradient1.ravel())
    plt.plot(gradient2.ravel())
    plt.plot(gradient3.ravel())
    print(np.linalg.norm(gradient1-gradient3))
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
    eps = 1.0e-3
    xa = np.linspace(params.lmin-eps, params.lc1-eps**2, 1000)
    repel = np.zeros_like(xa)
    repel = params.b * np.exp(1/(xa-params.lc1))/(xa-params.lmin)

    # attractive
    xb = np.linspace(params.lc0+eps**2, params.lmax+eps, 1000)
    attr = np.zeros_like(xb)
    attr = params.b * np.exp(1/(params.lc0-xb))/(params.lmax-xb)

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

def plot_mod_tether():
    """Plot tether potential."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.b = 100.0
    params.r = 2
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    print(params.lc0, params.lmax)
    print(params.lc1, params.lmin)

    # repelling part
    xa = np.linspace(0+0.0001, params.lc1-0.0001, 1000)
    repel = np.zeros_like(xa)
    repel = params.b * np.exp(xa/(xa-params.lc1))*xa**(-params.r)

    # attractive
    xb = np.linspace(params.lc0+0.0001, 4*a, 1000)
    attr = np.zeros_like(xb)
    attr = params.b * params.r**(params.r+1)*(xb-params.lc0)**(params.r)

    plt.plot(xa,repel)
    plt.plot(xb, attr)
    plt.axvline(0.0, color="r", lw=0.3, label="min")
    plt.axvline(params.lc0, lw=0.3, label="lc0")
    plt.axvline(params.lc1, lw=0.3, label="lc1")
    plt.axvline(a, color="k", lw=0.3, label="a")
    plt.title("Modified Noguchi tether potential")
    plt.ylim([-10, 500])
    plt.legend()
    plt.show()

def test_integration():
    """Test md."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.type = "tether"
    params.b = 100.0
    params.r = 2
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 4*a #1.33*a
    params.lmin = 0.0 #0.67*a

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.8, 1.0, 1.0, params)
    estore.init(mesh, ref_lambda=1.0)

    estore.print_info("")

    sigma = 1 #-> m
    gamma = 1.0

    x = mesh.points()
    v = np.random.normal(size=x.shape)*np.sqrt(sigma)

    def force(x,v):
        g = np.empty_like(x)
        m.gradient(mesh, estore, g, 1.0e-6)
        return -g-gamma*v

    xn, vn = vv(x, v, force, sigma, 0.001, 800)

    for i,xi in enumerate(xn):
        np.copyto(x,xi)
        om.write_mesh("out/test_"+str(i)+".stl", mesh)

    m.energy(mesh, estore)
    estore.print_info("")

def test_hmc():
    """Test hamilton monte carlo."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.type = "tether"
    params.b = 100
    params.r = 2
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 4*a #1.33*a
    params.lmin = 0.0 #0.67*a

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.8, 1.0, 1.0, params)
    estore.init(mesh, ref_lambda=1.0)

    estore.print_info("")

    sigma = 1 #-> m

    x0 = mesh.points().copy()

    def hamiltonian(x,v,T):
        points = mesh.points()
        np.copyto(points, x)
        vr = v.ravel()
        e = m.energy(mesh, estore)/T + 0.5*vr.dot(vr)/sigma
        np.copyto(points,x0)
        return e

    def force(x,v,T):
        points = mesh.points()
        np.copyto(points,x)
        g = np.empty_like(x)
        m.gradient(mesh, estore, g, 1.0e-6)
        np.copyto(points,x0)
        return -g/T

    def print_info(i,x,v,T,acc):
        print("\n-- Step ",i)
        print("  ----- Temperature:", T)
        print("  ----- acc-rate:   ", acc)
        p = mesh.points()
        np.copyto(p,x)
        m.energy(mesh, estore)
        estore.print_info("  ")

    def flip():
        m.flip_edges(mesh)

    xn = hmc(x0, hamiltonian, force, sigma, 4000, 0.001, 10, info=print_info)

    for i,xi in enumerate(xn):
        if i%10 == 0:
          x = mesh.points()
          np.copyto(x,xi)
          om.write_mesh("out/test_"+str(i)+".stl", mesh)

def test_minimization():
    """try direct minimization."""

    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    a = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.type = "tether"
    params.b = 100.0
    params.r = 2
    params.lc0 = 1.15*a
    params.lc1 = 0.85*a
    params.lmax = 10*a #1.33*a
    params.lmin = 0.0 #0.67*a

    estore = m.EnergyValueStore(1.0, 1.0e4, 1.0e4, 0.0, 0.8, 1.0, 1.0, params)
    estore.init(mesh, ref_lambda=1.0)
    estore.print_info("")

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
    res = minimize(fun, x0, jac=jac, options={"maxiter": 2000})
    print(res.nit, res.message)

    om.write_mesh("out/test0.stl", mesh)
    points = mesh.points()
    points += res.x.reshape(points.shape)
    om.write_mesh("out/test1.stl", mesh)

    estore.print_info("")

if __name__ == "__main__":
    #test_neighbourhood()
    #test_energy()
    #test_vv()
    #plot_tether()
    #plot_mod_tether()
    #test_integration()
    test_minimization()
    #test_hmc()
