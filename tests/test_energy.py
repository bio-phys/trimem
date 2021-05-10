import helfrich.openmesh as om
import helfrich.energy as pyenergy
import helfrich as cppenergy
import numpy as np
import meshzoo

import time


def sphere(radius, n):
    """Get sphere mesh with analytical reference values."""
    points, cells = meshzoo.icosa_sphere(n)
    tri = om.TriMesh(points*r, cells)
    print("\nGenerating sphere with {} vertices.".format(tri.n_vertices()))
    # mesh, energy/kappa, surf, vol, curv
    return tri, 8*np.pi, 4*np.pi*r**2, 4/3*np.pi*r**3, 4*np.pi*r

def tube(radius, n):
    """Get tube mesh with analytical reference values."""
    points, cells = meshzoo.tube(length=1, radius=radius, n=n)
    tri = om.TriMesh(points, cells)
    print("\nGenerating tube with {} vertices.".format(tri.n_vertices()))
    # mesh, energy/kappa, surf, vol, curv
    return tri, np.pi/r, 2*np.pi*r, 2/3*np.pi*r**2, np.pi

r = 1.0
n = 32

tri, e_ref, s_ref, v_ref, c_ref = sphere(r, n)
#tri, e_ref, s_ref, v_ref, c_ref = tube(r, n)

# write mesh for debugging
om.write_mesh("test.stl", tri)

# ---------------------------------------------------------------------------- #
# edge based evaluation in python
start = time.time()
for i in range(10):
    m,s,v,c = pyenergy.calc_energy(tri, 1.0)
dt = time.time()-start

print("\n-- edge-based (python) (wrong energy)")
print("Energy:    {} (={})".format(m, e_ref))
print("Surface:   {} (={})".format(s, s_ref))
print("Volume:    {} (={}".format(v, v_ref))
print("Curvature: {} (={})".format(c, c_ref))
print("time elapsed: {}".format(dt))

# ---------------------------------------------------------------------------- #
# vertex based evaluation in python
start = time.time()
for i in range(10):
    m,s,v,c = pyenergy.calc_energy_v(tri, 1.0)
dt = time.time()-start

print("\n-- vertex-based (python)")
print("Energy:    {} (={})".format(m, e_ref))
print("Surface:   {} (={})".format(s, s_ref))
print("Volume:    {} (={}".format(v, v_ref))
print("Curvature: {} (={})".format(c, c_ref))
print("time elapsed: {}".format(dt))

# ---------------------------------------------------------------------------- #
# edge based evaluation in c++
start = time.time()
for i in range(10):
    m,s,v,c = cppenergy.calc_properties_e(tri)
dt = time.time()-start

print("\n-- edge-based (c++) (wrong energy)")
print("Energy:    {} (={})".format(m, e_ref))
print("Surface:   {} (={})".format(s, s_ref))
print("Volume:    {} (={}".format(v, v_ref))
print("Curvature: {} (={})".format(c, c_ref))
print("time elapsed: {}".format(dt))

# ---------------------------------------------------------------------------- #
# vertex based evaluation in c++
start = time.time()
for i in range(10):
    m,s,v,c = cppenergy.calc_properties_v(tri)
dt = time.time()-start

print("\n-- vertex-based (c++)")
print("Energy:    {} (={})".format(m, e_ref))
print("Surface:   {} (={})".format(s, s_ref))
print("Volume:    {} (={}".format(v, v_ref))
print("Curvature: {} (={})".format(c, c_ref))
print("time elapsed: {}".format(dt))

# ---------------------------------------------------------------------------- #
# vertex based evaluation in c++ (version 2)
start = time.time()
for i in range(10):
    m,s,v,c = cppenergy.calc_properties_vv(tri)
dt = time.time()-start

print("\n-- vertex-based (c++) (version 2)")
print("Energy:    {} (={})".format(m, e_ref))
print("Surface:   {} (={})".format(s, s_ref))
print("Volume:    {} (={}".format(v, v_ref))
print("Curvature: {} (={})".format(c, c_ref))
print("time elapsed: {}".format(dt))
