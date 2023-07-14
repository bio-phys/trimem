
# Here we set up the TriSim class containing:
# eparams -> estore
# output_params
# algo_params
# methods: minim and mc
# the current Mesh

from .. import core as m
from trimem.core import TriMesh
from trimem.mc.mesh import Mesh
from .hmc import MeshHMC, MeshFlips, MeshMonteCarlo, get_step_counters
from collections import Counter
import concurrent.futures
import pickle
from copy import deepcopy




class InitialState():
    def __init__(self,mesh_points,mesh_faces,initialize):
        self.initialize = initialize
        self.state=Mesh(points=mesh_points,cells=mesh_faces)



class OutputParams():
    def __init__(self,
                 info,
                 thin,
                 out_every,
                 input_set,  # hast to be stl file or if None uses mesh
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format
                 ):
        self.info=info
        self.thin=thin
        self.out_every=out_every
        self.input_set=input_set
        self.output_prefix=output_prefix
        self.restart_prefix=restart_prefix
        self.checkpoint_every = checkpoint_every,
        self.output_format = output_format

class AlgoParams():
    def __init__(self,
                 num_steps,
                 init_step,
                 step_size,
                 traj_steps,
                 momentum_variance,
                 flip_ratio,
                 flip_type,
                 initial_temperature,
                 cooling_factor,
                 start_cooling,
                 maxiter):
        self.num_steps=num_steps
        self.init_step=init_step
        self.step_size=step_size
        self.traj_steps=traj_steps
        self.momentum_variance=momentum_variance
        self.flip_ratio=flip_ratio
        self.flip_type=flip_type
        self.initial_temperature=initial_temperature
        self.cooling_factor=cooling_factor
        self.start_cooling=start_cooling
        self.maxiter=maxiter

class TriSim():

    def __init__(self,
                 #Initialization (Determines if mesh is used as reference in estore)
                 initialize=True,
                 initial_point=None,
                 initial_faces=None,
                 #MESH
                 mesh_points=None,
                 mesh_faces=None,
                 #BOND
                 bond_type='Edge',
                 bond_r=1,
                 lc0=None,
                 lc1=None,
                 a0=None,
                 #SURFACEREPULSION
                 n_search="cell-list",
                 rlist=0.1,
                 exclusion_level=2,
                 rep_lc1=0.0,
                 rep_r= 2,
                 delta= 0.0,
                 lam= 1.0,
                 kappa_b = 30.0,
                 kappa_a = 1.0e6,
                 kappa_v = 1.0e6,
                 kappa_c = 1.0e6,
                 kappa_t = 1.0e5,
                 kappa_r = 1.0e3,
                 area_frac = 1.0,
                 volume_frac = 1.0,
                 curvature_frac = 1.0,

                 #ALGORITHM
                 num_steps=10,
                 init_step='{}',
                 step_size=7e-5,
                 traj_steps=100,
                 momentum_variance=1.0,
                 flip_ratio=0.1,
                 flip_type='parallel',
                 initial_temperature=1.0,
                 cooling_factor=1.0e-4,
                 start_cooling=0,
                 maxiter=10,

                 #OUTPUT
                 info=10,
                 thin=10,
                 out_every=0,
                 input_set='inp.stl',  # hast to be stl file or if True uses mesh
                 output_prefix='inp',
                 restart_prefix='inp',
                 checkpoint_every=0,
                 output_format='xyz'


                 ):

        self._bond_enums = {
            "Edge": m.BondType.Edge,
            "Area": m.BondType.Area
        }
        # Argument: mesh should be Mesh object gets converted to Mesh.trimesh (TriMesh) internally

        self.mesh=Mesh(points=mesh_points,cells=mesh_faces)
        #self.mesh.trimesh=m.TriMesh(mesh_points,mesh_faces)

        if initialize:
            self.initialstate=InitialState(mesh_points,mesh_faces,initialize)
            self.initialstate.initialize=False
        else:
            self.initialstate=InitialState(initial_point,initial_faces,initialize)

        self.bparams = m.BondParams()
        if issubclass(type(bond_type), str):
            self.bparams.type = self._bond_enums[bond_type]
        else:
            self.bparams.type=bond_type
        self.bparams.r = bond_r

        if (lc1 is None) and (lc0 is None):
            a, l = m.avg_tri_props(self.mesh.trimesh)
            self.bparams.lc0 = 1.15 * l
            self.bparams.lc1 = 0.85 * l
            self.bparams.a0 = a
        else:
            self.bparams.lc0 = lc0
            self.bparams.lc1 = lc1
            self.bparams.a0  = a0

        self.rparams = m.SurfaceRepulsionParams()
        self.rparams.n_search = n_search
        self.rparams.rlist = rlist
        self.rparams.exclusion_level = exclusion_level
        self.rparams.lc1 = rep_lc1
        self.rparams.r = rep_r

            # translate energy params

        self.cp = m.ContinuationParams()
        self.cp.delta = delta
        self.cp.lam = lam

        self.eparams = m.EnergyParams()
        self.eparams.kappa_b = kappa_b
        self.eparams.kappa_a = kappa_a
        self.eparams.kappa_v = kappa_v
        self.eparams.kappa_c = kappa_c
        self.eparams.kappa_t = kappa_t
        self.eparams.kappa_r = kappa_r
        self.eparams.area_frac = area_frac
        self.eparams.volume_frac = volume_frac
        self.eparams.curvature_frac = curvature_frac
        self.eparams.bond_params = self.bparams
        self.eparams.repulse_params = self.rparams
        self.eparams.continuation_params = self.cp

        self.algo_params=AlgoParams(num_steps,init_step,step_size,traj_steps,
                 momentum_variance,flip_ratio,flip_type,initial_temperature,
                 cooling_factor,start_cooling,maxiter)


        self.output_params=OutputParams(info,
                 thin,
                 out_every,
                 input_set,
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format)

        self.estore = m.EnergyManager(self.initialstate.state.trimesh, self.eparams)

        #self.flips = MeshFlips(mesh, estore, options={
       #     "flip_type": cmc["flip_type"],
       #     "flip_ratio": cmc.getfloat("flip_ratio"),
      #      "info_step": config["GENERAL"].getint("info"),
      #  })


    def __reduce__(self):
        return self.__class__,(self.initialstate.initialize,self.initialstate.state.x,self.initialstate.state.f,
                 self.mesh.x,
                 self.mesh.f,
                 self.estore.eparams.bond_params.type,
                 self.estore.eparams.bond_params.r,
                 self.estore.eparams.bond_params.lc0,
                 self.estore.eparams.bond_params.lc1,
                 self.estore.eparams.bond_params.a0,
                 self.estore.eparams.repulse_params.n_search,
                 self.estore.eparams.repulse_params.rlist,
                 self.estore.eparams.repulse_params.exclusion_level,
                 self.estore.eparams.repulse_params.lc1,
                 self.estore.eparams.repulse_params.r,
                 self.estore.eparams.continuation_params.delta,
                 self.estore.eparams.continuation_params.lam,
                 self.estore.eparams.kappa_b ,
                 self.estore.eparams.kappa_a  ,
                 self.estore.eparams.kappa_v  ,
                 self.estore.eparams.kappa_c ,
                 self.estore.eparams.kappa_t,
                 self.estore.eparams.kappa_r ,
                 self.estore.eparams.area_frac ,
                 self.estore.eparams.volume_frac ,
                 self.estore.eparams.curvature_frac,
                 self.algo_params.num_steps,
                 self.algo_params.init_step,
                 self.algo_params.step_size,
                 self.algo_params.traj_steps,
                 self.algo_params.momentum_variance,
                 self.algo_params.flip_ratio,
                 self.algo_params.flip_type,
                 self.algo_params.initial_temperature,
                 self.algo_params.cooling_factor,
                 self.algo_params.start_cooling,
                 self.algo_params.maxiter,

                 self.output_params.info,
                 self.output_params.thin,
                 self.output_params.out_every,
                 self.output_params.input_set,  # hast to be stl file or if True uses mesh
                 self.output_params.output_prefix,
                 self.output_params.restart_prefix,
                 self.output_params.checkpoint_every,
                 self.output_params.output_format)


    def update_energy_manager(self):
        self.estore = m.EnergyManager(self.initialstate.state.trimesh, self.eparams)

    def update_energy_parameters(self):
        self.eparams = self.estore.eparams


    def subprocess(self):
        self.eparams.kappa_r-=1
        self.update_energy_manager()
        return self


def master_process(trisim):


    for i_p in range(1):
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(trisim.subprocess)
            trisim = future.result()

    return trisim









