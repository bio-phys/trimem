
# Here we set up the TriSim class containing:
# eparams -> estore
# output_params
# algo_params
# methods: minim and mc
# the current Mesh

import warnings
from datetime import datetime, timedelta
import psutil

from .. import core as m
from trimem.core import TriMesh
from trimem.mc.mesh import Mesh
#from .evaluators import PerformanceEnergyEvaluators,  TimingEnergyEvaluators
from .hmc import MeshHMC, MeshFlips, MeshMonteCarlo, get_step_counters
from trimem.mc.output import make_output
from collections import Counter
import concurrent.futures
import pickle
import pathlib
from copy import deepcopy
import numpy as np
import time
from scipy.optimize import minimize



class Timer():
    "Storage for timer state to reinitialize PerformanceEnergyEvaluator after Reset"
    def __init__(self,ptime,ts,ta,tan,ts_default,stime):
        self.performance_start=ptime
        self.performance_timestamps=ts
        self.timearray=ta
        self.timearray_new=tan
        self.timestamps=ts_default
        self.start=stime


class InitialState():
    def __init__(self,area,volume,curvature,bending,tethering):
        self.area=area
        self.volume=volume
        self.curvature=curvature
        self.bending=bending
        self.tethering=tethering
       # self.repulsion=repulsion




class OutputParams():
    def __init__(self,
                 info,
                 thin,
                 out_every,
                 input_set,  # hast to be stl file or if None uses mesh
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format,
                 output_flag,
                 output_counter,
                 performance_increment,
                 energy_increment
                 ):
        self.info=info
        self.thin=thin
        self.out_every=out_every
        self.input_set=input_set
        self.output_prefix=output_prefix
        self.restart_prefix=restart_prefix
        self.checkpoint_every = checkpoint_every
        self.output_format = output_format
        self.output_flag = output_flag
        self.output_counter=output_counter
        self.performance_increment=performance_increment
        self.energy_increment=energy_increment

class AlgoParams():
    def __init__(self,
                 num_steps,
                 reinitialize_every,
                 init_step,
                 step_size,
                 traj_steps,
                 momentum_variance,
                 flip_ratio,
                 flip_type,
                 initial_temperature,
                 cooling_factor,
                 start_cooling,
                 maxiter,
                 refresh):
        self.num_steps=num_steps
        self.reinitialize_every=reinitialize_every
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
        self.refresh=refresh

class TriSim():



    def __init__(self,
                 #Initialization (Determines if mesh is used as reference in estore)
                 initialize=True,
                 #MESH
                 mesh_points=None,
                 mesh_faces=None,
                 #BOND
                 bond_type='Edge',
                 bond_r=2,
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
                 reinitialize_every=10000,
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
                 refresh=1,

                 #OUTPUT
                 info=10,
                 thin=10,
                 out_every= 0,
                 input_set='inp.stl',  # hast to be stl file or if True uses mesh
                 output_prefix='inp',
                 restart_prefix='inp',
                 checkpoint_every= 1,
                 output_format='xyz',
                 output_flag='A',
                 output_counter=0,
                 performance_increment=1000,
                 energy_increment=250,

                 area=1.0,
                 volume=1.0,
                 curvature=1.0,
                 bending=1.0,
                 tethering=1.0,
                 #repulsion=1.0,

                 ptime=time.time(),
                 ptimestamp=[],
                 dtimestamp=[],
                 timearray=np.zeros(2),
                 timearray_new=np.zeros(2),
                 stime=datetime.now(),

                 move_count=0,
                 flip_count=0



                 ):

        self.initialize = initialize

        self.flatten = True
        if self.flatten:
            self._ravel = lambda x: np.ravel(x)
        else:
            self._ravel = lambda x: x


        self._bond_enums = {
            "Edge": m.BondType.Edge,
            "Area": m.BondType.Area
        }
        # Argument: mesh should be Mesh object gets converted to Mesh.trimesh (TriMesh) internally

        self.mesh=Mesh(points=mesh_points,cells=mesh_faces)




        self.bparams = m.BondParams()
        if issubclass(type(bond_type), str):
            self.bparams.type = self._bond_enums[bond_type]
        else:
            self.bparams.type=bond_type
        self.bparams.r = bond_r

        if (lc1 is None) and (lc0 is None):
            a, l = m.avg_tri_props(self.mesh.trimesh)
            self.bparams.lc0 = 1.25 * l
            self.bparams.lc1 = 0.75 * l
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
        #self.rparams.lc1 = l*0.001
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

        self.algo_params=AlgoParams(num_steps,reinitialize_every,init_step,step_size,traj_steps,
                 momentum_variance,flip_ratio,flip_type,initial_temperature,
                 cooling_factor,start_cooling,maxiter,refresh)


        self.output_params=OutputParams(info,
                 thin,
                 out_every,
                 input_set,
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format,
                output_flag,
                output_counter,
                performance_increment,
                energy_increment)

        if self.initialize:
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams)
            self.initial_state=InitialState(self.estore.initial_props.area,
                                            self.estore.initial_props.volume,
                                            self.estore.initial_props.curvature,
                                            self.estore.initial_props.bending,
                                            self.estore.initial_props.tethering)
                                            #self.estore.initial_props.repulsion)



            self.initialize=False




        else:
            self.initial_state = InitialState(area,
                                              volume,
                                              curvature,
                                              bending,
                                              tethering
                                              )
            self.init_props = m.VertexPropertiesNSR()
            self.init_props.area = self.initial_state.area
            self.init_props.volume = self.initial_state.volume
            self.init_props.curvature = self.initial_state.curvature
            self.init_props.bending = self.initial_state.bending
            self.init_props.tethering = self.initial_state.tethering
           # self.init_props.repulsion = self.initial_state.repulsion



            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams, self.init_props)


        # BOOKKEEPING
        self.counter = Counter(move=move_count, flip=flip_count)
        self.timer = Timer(ptime, ptimestamp, timearray, timearray_new,dtimestamp,stime)
        self.output = make_output(self.output_params.output_format, self.output_params.output_prefix,self.output_params.output_counter,callback=self.update_output_counter)
        self.cpt_writer = self.make_checkpoint_handle()
        self.process=psutil.Process()
        self.n = self.algo_params.num_steps // self.output_params.info


        # FLIP/EVALUATOR/HMC/MMC - SETUp
        self.flips = MeshFlips(self.mesh, self.estore, options={
            "flip_type" : self.algo_params.flip_type,
            "flip_ratio" : self.algo_params.flip_ratio,
            "info_step" : self.output_params.info
        })


        #self.funcs = PerformanceEnergyEvaluators(self.mesh, self.estore, self.output, {
        #   "info_step": self.output_params.info,
        #    "output_step": self.output_params.thin,
        #    "cpt_step": self.output_params.checkpoint_every,
        #    "refresh_step": self.algo_params.refresh,
        #    "num_steps": self.algo_params.num_steps,
        #    "write_cpt": self.cpt_writer,
        #    "prefix": self.output_params.output_prefix
        #}, self.timer)

        self.hmc = MeshHMC(self.mesh, self.fun, self.grad, options={
            "mass": self.algo_params.momentum_variance,
            "time_step": self.algo_params.step_size,
            "num_integration_steps": self.algo_params.traj_steps,
            "initial_temperature": self.algo_params.initial_temperature,
            "cooling_factor": self.algo_params.cooling_factor,
            "cooling_start_step": self.algo_params.start_cooling,
            "info_step": self.output_params.info,
        }, counter=self.counter)

        self.mmc = MeshMonteCarlo(self.hmc, self.flips, self.timer.timearray_new, counter=self.counter, callback=self.callback,
                             extra_callback=self.extra_callback)

        self.info_step = max(self.output_params.info, 0)
        self.out_step = max(self.output_params.thin, 0)
        self.cpt_step = max(self.output_params.checkpoint_every, 0)
        self.refresh_step = max(self.algo_params.refresh, 0)




    def __reduce__(self):
        return self.__class__,(self.initialize,
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
                 self.estore.eparams.kappa_b,
                 self.estore.eparams.kappa_a,
                 self.estore.eparams.kappa_v,
                 self.estore.eparams.kappa_c,
                 self.estore.eparams.kappa_t,
                 self.estore.eparams.kappa_r,
                 self.estore.eparams.area_frac,
                 self.estore.eparams.volume_frac,
                 self.estore.eparams.curvature_frac,
                 self.algo_params.num_steps,
                 self.algo_params.reinitialize_every,
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
                 self.algo_params.refresh,

                 self.output_params.info,
                 self.output_params.thin,
                 self.output_params.out_every,
                 self.output_params.input_set,  # hast to be stl file or if True uses mesh
                 self.output_params.output_prefix,
                 self.output_params.restart_prefix,
                 self.output_params.checkpoint_every,
                 self.output_params.output_format,
                 self.output_params.output_flag,
                 self.output_params.output_counter,
                 self.output_params.performance_increment,
                 self.output_params.energy_increment,


                 self.estore.initial_props.area,
                 self.estore.initial_props.volume,
                 self.estore.initial_props.curvature,
                 self.estore.initial_props.bending,
                 self.estore.initial_props.tethering,
                 #self.estore.initial_props.repulsion,
                 self.timer.performance_start,
                 self.timer.performance_timestamps,
                 self.timer.timestamps,
                 self.timer.timearray,
                 self.timer.timearray_new,
                 self.timer.start,
                 self.counter["move"],
                 self.counter["flip"]
                               )

    def make_checkpoint_handle(self):

        return self.make_checkpoint

    def make_checkpoint(self, force_name=None):

        if not force_name:

            cptfname = pathlib.Path(self.output_params.output_prefix)
            cptfname = cptfname.name + self.output_params.output_flag + '.cpt'

            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.output_params.output_flag == 'A':
                self.output_params.output_flag = 'B'
            else:
                self.output_params.output_flag = 'A'
        else:
            cptfname = pathlib.Path(force_name)
            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'made cp:{cptfname}')



    def minim(self):
        """Run (precursor) minimization.

        Performs a minimization of the Helfrich bending energy as defined
        by the `config`.

        Args:
            mesh (mesh.Mesh): initial geometry.
            estore (EnergyManager): EnergyManager.
            config (dict-like): run-config file.


        """
        refresh_safe = self.algo_params.refresh



        if not self.algo_params.refresh == 1:
            wstr = f"SURFACEREPULSION::refresh is set to {self.algo_params.refresh}, " + \
                   "which is ignored in in minimization."
            warnings.warn(wstr)

            self.algo_params.refresh = 1

        step_count = Counter(move=0, flip=0)

        def _cb(x):
            self.callback(x, step_count)
            step_count["move"] += 1

        # run minimization
        options = {
            "maxiter": self.algo_params.maxiter,
            "disp": 0,


        }
        res = minimize(
            self.fun,
            self._ravel(self.mesh.x),
            #self.mesh.x,
            jac=self.grad,
            callback=_cb,
            method="L-BFGS-B",
            options=options
        )
        self.mesh.x = res.x.reshape(self.mesh.x.shape)
        self.algo_params.refresh=refresh_safe

        # print info
        print("\n-- Minimization finished at iteration", res.nit)
        print(res.message)
        self.estore.print_info(self.mesh.trimesh)

        # write final checkpoint
        self.cpt_writer()
        self.reset_counter()
        #self.reset_output_counter()

    def _update_mesh(func):
        """Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
        return wrap

    @_update_mesh
    def fun(self, x):
        """Evaluate energy.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.energy(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            float:
                Value of the Energy represented by ``self.estore``.
        """
        return self._ravel(self.estore.energy(self.mesh.trimesh))

    @_update_mesh
    def grad(self, x):
        """Evaluate gradient.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.gradient(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            ndarray[float]:
                Gradient with respect to `x` of the Energy represented by
                ``self.estore``.
        """
        return self._ravel(self.estore.gradient(self.mesh.trimesh))

    @_update_mesh
    def grad_unraveled(self, x):
        """Evaluate gradient.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.gradient(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            ndarray[float]:
                Gradient with respect to `x` of the Energy represented by
                ``self.estore``.
        """
        return self.estore.gradient(self.mesh.trimesh)

    @_update_mesh
    def callback(self, x, steps):




        """Callback.

        Allows for the injection of custom trimem functionality into generic
        sampling and minimization algorithms:

            * stdout verbosity
            * writing of output trajectories
            * writing of checkpoint files
            * update of the internal state of self.estore

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in self.mesh.
            steps (collections.Counter): step counter dictionary
            args: ignored

        Keyword Args:
            kwargs: ignored
        """
        i = sum(steps.values()) #py3.10: steps.total()

        if self.output_params.info and (i % self.output_params.info == 0):
            print("\n-- Energy-Evaluation-Step ", i)
            self.estore.print_info(self.mesh.trimesh)
        if self.output_params.thin and (i % self.output_params.thin == 0):
            self.output.write_points_cells(self.mesh.x, self.mesh.f)
        if self.output_params.checkpoint_every and (i % self.output_params.checkpoint_every == 0):
            self.cpt_writer()
       # if self.algo_params.refresh and (i % self.algo_params.refresh == 0):
        #    self.estore.update_repulsion(self.mesh.trimesh)
        self.estore.update_reference_properties()

        if self.output_params.energy_increment and (i % self.output_params.energy_increment==0):
            with open(f'energies_vol{self.estore.eparams.volume_frac*100:03.0f}_cur{self.estore.eparams.curvature_frac*100:03.0f}.dat','a+') as f:
                f.write(f'{i} {self.estore.energy(self.mesh.trimesh)}\n')

        if self.output_params.info and (i % self.output_params.info == 0):
            self.timer.timestamps.append(time.time())
            if len(self.timer.timestamps) == 2:
                tspan = self.timer.timestamps[1] - self.timer.timestamps[0]
                speed = tspan / self.output_params.info
                finish = self.timer.start + timedelta(seconds=tspan) * self.n
                print("\n-- Performance measurements")
                print(f"----- estimated speed: {speed:.3e} s/step")
                print(f"----- estimated end:   {finish}")
                self.timer.timestamps.pop(0)

            # Section for the preformance measurement of the code
        if i == 1:
            with open(f'{self.output_params.output_prefix}_performance.dat', 'w') as file:
                file.write(
                    '#Step Elapsed_Time Time_Per_Step %Vertex_Moves %Mesh_Flips %Residue %CPU RAM_USAGE %RAM RAM_AVAILABLE_PRC RAM_TOTAL\n')
                # tracemalloc.start()

        if (i % self.output_params.performance_increment == 0):
            self.timer.performance_timestamps.append(time.time())
            section_time = self.timer.timearray_new - self.timer.timearray
            self.timer.timearray = self.timer.timearray_new.copy()
            self.process = psutil.Process()

            if len(self.timer.performance_timestamps) == 2:
                performance_tspan = self.timer.performance_timestamps[1] - self.timer.performance_timestamps[0]

                with open(f'{self.output_params.output_prefix}_performance.dat', 'a') as file:
                    file.write(f'{i} {self.timer.performance_timestamps[1] - self.timer.performance_start:.4f}'
                               f' {performance_tspan / self.output_params.performance_increment:.4f}'
                               f' {section_time[0] / performance_tspan:.4f} {section_time[1] / performance_tspan:.4f}'
                               f' {(performance_tspan - section_time[0] - section_time[1]) / performance_tspan:.4f}'
                               f' {self.process.cpu_percent(interval=None):.4f} {self.process.memory_info().vms / 1024 ** 3:.4f}'
                               f' {self.process.memory_percent(memtype="vms"):.4f} {psutil.virtual_memory()[1] / 1000000000:.4f}'
                               f' {psutil.virtual_memory()[0] / 1000000000:.4f}\n'
                               )

                self.timer.performance_timestamps.pop(0)

    def extra_callback(self, timearray_loc):
        self.timer.timearray_new=timearray_loc



    def update_energy_manager(self):
        self.estore = m.EnergyManager(self.mesh.trimesh, self.eparams)

    def update_energy_parameters(self):
        self.eparams = self.estore.eparams


    def reset_counter(self,move=0,flip=0):

        self.counter = Counter(move=move, flip=flip)
        self.mmc.counter = self.counter
        self.hmc.counter = self.counter
        self.flips.counter = self.counter



    def reset_output_counter(self):
        self.output_params.output_counter=0

    def update_output_counter(self,ocn):
        self.output_params.output_counter = ocn

    def update_output(self):
        self.output = make_output(self.output_params.output_format, self.output_params.output_prefix,
                                  self.output_params.output_counter, callback=self.update_output_counter)

    def subprocess(self):
        print('made it into a subprocess')
        self.mmc.run(self.algo_params.reinitialize_every)
        self.mesh.x=self.hmc.mesh.x
        print('made it through a subprocess')
        return self

    def run(self):
        self.mmc.run(self.algo_params.num_steps)
        self.mesh.x=self.hmc.mesh.x


def master_process(trisim):




    for i_p in range(np.int64(trisim.algo_params.num_steps/trisim.algo_params.reinitialize_every)):

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(trisim.subprocess)
            trisim = future.result()

    return trisim

def read_checkpoint(fname):

    with open(fname, 'rb') as f:
        trisim=pickle.load(f)
    return trisim

