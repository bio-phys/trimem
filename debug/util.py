"""
Utilities for debugging.
"""
import helfrich as m
import numpy as np

def get_energy_manager(mesh, bond_type, kb, ka, kv, kc, kt, af=1., vf=1., cf=1.):
    """Setup energy manager."""
    l = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    a = m.area(mesh)/mesh.n_faces();

    bparams = m.BondParams()
    bparams.type = bond_type
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.r    = 2
    bparams.a0   = a

    eparams = m.EnergyParams()
    eparams.kappa_b        = kb
    eparams.kappa_a        = ka
    eparams.kappa_v        = kv
    eparams.kappa_c        = kc
    eparams.kappa_t        = kt
    eparams.area_frac      = af
    eparams.volume_frac    = vf
    eparams.curvature_frac = cf
    eparams.bond_params    = bparams

    return m.EnergyManager(mesh, eparams)

