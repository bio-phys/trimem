import helfrich.openmesh as om
import helfrich.energy as pyenergy
import helfrich.test as cppenergy
import numpy as np
import meshzoo

import time

# unit sphere
points, cells = meshzoo.icosa_sphere(32)
tri = om.TriMesh(points, cells)
om.write_mesh("test.stl", tri)
print(tri.n_vertices())

m, s, v, c = pyenergy.calc_energy(tri, 1.0)

start = time.time()
for i in range(10):
    m,s,v,c = pyenergy.calc_energy(tri, 1.0)
dt = time.time()-start

print("--")
print("Energy:", m)
print("Surface: {} (4*pi={})".format(s, 4*np.pi))
print("Volume: {} (4/3*pi={}".format(v, 4/3*np.pi))
print("Curvature: {} (4*pi={})".format(c, 4*np.pi))
print("time elapsed: {}".format(dt))


m, s, v, c = pyenergy.calc_energy(tri, 1.0)

start = time.time()
for i in range(10):
    m = cppenergy.print_mesh(tri, 1.0)
dt = time.time()-start

print("--")
print("Energy:", m)
#print("Surface: {} (4*pi={})".format(s, 4*np.pi))
#print("Volume: {} (4/3*pi={}".format(v, 4/3*np.pi))
#print("Curvature: {} (4*pi={})".format(c, 4*np.pi))
print("time elapsed: {}".format(dt))
