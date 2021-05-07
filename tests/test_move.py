import helfrich.openmesh as om
import helfrich.test as test
import meshzoo
import numpy as np


points, cells = meshzoo.icosa_sphere(16)
tri = om.TriMesh(points, cells)

e = test.EnergyValueStore(1.0, 1.0, 1.0, 1.0, 1.0)
e.init(tri)

print(e.get_energy())

om.write_mesh("test0.stl", tri)
idx = np.random.choice(tri.n_vertices(), tri.n_vertices(), replace=False)
acc = test.mc_move_serial(tri, e, idx, 0.01, 0.01, 0.2)
print(acc/tri.n_vertices())

print(e.get_energy())
om.write_mesh("test1.stl", tri)
