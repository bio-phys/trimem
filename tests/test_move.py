import helfrich.openmesh as om
import helfrich as test
import meshzoo
import numpy as np


points, cells = meshzoo.icosa_sphere(16)
tri = om.TriMesh(points, cells)
n = tri.n_vertices()
om.write_mesh("test0.stl", tri)

e = test.EnergyValueStore(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
e.init(tri)
print(e.get_energy())

c = test.MeshConstraintNL(tri, 0.01, 1.0)


idx  = np.random.choice(n, n, replace=False)
vals = np.random.uniform(-0.005,0.005, size=(n,3))
acc = test.move_serial(tri, e, idx, vals, 0.01, 0.3, c, 1.0)
print(acc)

print(e.get_energy())
om.write_mesh("test1.stl", tri)
