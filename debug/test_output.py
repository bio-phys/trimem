from helfrich.mc.output.xdmf import XdmfWriter
from helfrich.mc.output.xyz import XyzWriter
from helfrich.mc.output.vtu import VtuWriter
from helfrich.mc.output.util import create_backup
import meshzoo
import numpy as np


xdmf = XdmfWriter("tmp")
xyz  = XyzWriter("tmp")
vtu  = VtuWriter("tmp")

# step 1
p,c = meshzoo.icosa_sphere(4)
xdmf.write_points_cells(p,c)
xyz.write_points_cells(p,c)
vtu.write_points_cells(p,c)

# step 2 with different mesh size
p,c = meshzoo.icosa_sphere(6)
xdmf.write_points_cells(p,c)
xyz.write_points_cells(p,c)
vtu.write_points_cells(p,c)

create_backup("tmp")


