"""
Utilities for debugging.
"""
import helfrich as m
import numpy as np

def get_energy_manager(mesh, bond_type, kb, ka, kv, kc, kt, kr, af=1, vf=1, cf=1):
    """Setup energy manager."""
    a,l = m.avg_tri_props(mesh)

    bparams = m.BondParams()
    bparams.type = bond_type
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.r    = 2
    bparams.a0   = a

    rparams = m.SurfaceRepulsionParams()
    rparams.rlist    = 0.5
    rparams.n_search = "verlet-list"
    rparams.lc1      = 0.25
    rparams.r        = 2

    eparams = m.EnergyParams()
    eparams.kappa_b        = kb
    eparams.kappa_a        = ka
    eparams.kappa_v        = kv
    eparams.kappa_c        = kc
    eparams.kappa_t        = kt
    eparams.kappa_r        = kr
    eparams.area_frac      = af
    eparams.volume_frac    = vf
    eparams.curvature_frac = cf
    eparams.bond_params    = bparams
    eparams.repulse_params = rparams

    return m.EnergyManager(mesh, eparams)

