import helfrich.openmesh as om
import helfrich.test as test
import meshzoo
import numpy as np


points, cells = meshzoo.icosa_sphere(16)
tri = om.TriMesh(points, cells)

print(test.volume_v(tri))
print(test.volume_f(tri))
