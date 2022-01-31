from helfrich.mc.output.xdmf import XdmfWriter
from helfrich.mc.output.xyz import XyzWriter
from helfrich.mc.output.vtu import VtuWriter
from helfrich.mc.output.util import create_backup
from helfrich.mc.output.checkpoint import CheckpointWriter, CheckpointReader
from helfrich.mc.config import CONF
import meshzoo
import numpy as np
import configparser


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

cpt = CheckpointWriter("tmp")
conf = configparser.ConfigParser()
conf.read_string(CONF)
cpt.write(p,c,conf)

cpt = CheckpointReader("tmp", 0)
p_in, c_in, conf_in = cpt.read()

create_backup("tmp", "tmp")
