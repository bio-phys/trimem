import warnings
from datetime import datetime, timedelta
import psutil

from copy import copy

import trimesh
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


from ctypes import *
from lammps import lammps, PyLammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY, LMP_SIZE_VECTOR, LMP_SIZE_ROWS, LMP_SIZE_COLS





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

class TriLmp():



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



        ## trimeshes
        self.mesh = Mesh(points=mesh_points, cells=mesh_faces)
        self.mesh_temp = Mesh(points=mesh_points, cells=mesh_faces)
        self.n_vertices=self.mesh.x.shape[0]

        ##beads








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
            # setup energy manager with initial mesh
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams)
            #safe initial states property
            self.initial_state=InitialState(self.estore.initial_props.area,
                                            self.estore.initial_props.volume,
                                            self.estore.initial_props.curvature,
                                            self.estore.initial_props.bending,
                                            self.estore.initial_props.tethering)
                                            #self.estore.initial_props.repulsion)






            self.initialize=False







        else:
            # reinitialize using saved initial state properties (for reference potential V, A, dA)
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

            #recreate energy manager
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams, self.init_props)



    #   SETTING UP LAMMPS
        # create internal lammps instance
        self.lmp = lammps()
        self.L = PyLammps(ptr=self.lmp,verbose=True)
        # use basic setup for system
        basic_system = f"""              units lj
                                        dimension 3
                                        
                                        
                                        atom_style	hybrid bond charge
                                        
                                        atom_modify sort 0 0.0
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        region box block -5 5 -5 5 -5 5
                                        create_box 1 box bond/types 1 extra/bond/per/atom 10 extra/special/per/atom 10
                                         
                                        run_style verlet
                                        thermo_modify lost ignore norm no

                                        fix 1 all  nve/limit 2.0
                                        fix ext all external pf/callback 1 1
                                        fix_modify ext energy yes
                                        timestep 0.00007
                                        
                                        special_bonds coul 0.0 0.0 0.0
                                        bond_style zero nocoeff
                                        bond_coeff 1 1 0.0
                                        
                                        
                                        pair_style coul/debye {1/(0.5*l)} {1.5*l}
                                        pair_coeff * *
                                        pair_coeff 1 1 {1.5*l}
                                        
                                        mass            1 1.0
                                        dielectric  1.0
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        """
        #pair_style
        #zero
        #1
        #nocoeff
        # pair_coeff 1 1
        #thermo 1000
        #


        # initial verlocities


        self.atom_props = f"""     
                
                
                velocity        all create {self.algo_params.initial_temperature} 1298371 mom yes dist gaussian   

                """

        # set vertex atomtype to 1
       # atype = np.ndarray.tolist(np.ones((self.mesh.x.shape[0]), dtype=np.int64))

        # initialize lammps
        self.lmp.commands_string(basic_system)

        # create vertices
        #self.lmp.create_atoms(self.mesh.x.shape[0], range(1,self.mesh.x.shape[0]+1), atype, np.ravel(self.mesh.x));
       # self.x0 = self.lmp.numpy.extract_atom("x")[0].copy()

        # set random thermal velocities


        #set initial bond topology from mesh
        self.edges = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f).edges_unique
        self.edges=np.unique(self.edges,axis=0)
        print('yo')
        with open('bonds.in','w') as f:
            f.write('\n\n')
            f.write(f'{self.mesh.x.shape[0]} atoms\n')
            f.write(f'{self.edges.shape[0]} bonds\n\n')

            f.write(f'1 atom types\n')
            f.write(f'1 bond types\n\n')

            f.write(f'Atoms #bond\n\n')
            for i in range(self.mesh.x.shape[0]):
                f.write(f'{i+1} 1  {self.mesh.x[i,0]} {self.mesh.x[i,1]} {self.mesh.x[i,2]} 1 1.0 \n')

            #f.write(f'{self.mesh.x.shape[0]+1} {self.mesh.x[0,0]} {self.mesh.x[0,1]} {self.mesh.x[0,2]} 1 2\n')
            f.write(f'Bonds #zero\n\n')

            for i in range(self.edges.shape[0]):
                f.write(f'{i+1} 1 {self.edges[i, 0] + 1} {self.edges[i, 1] + 1}\n')


        self.lmp.command('read_data bonds.in add merge')
        self.lmp.commands_string(self.atom_props)

                #self.lmp.command(f'create_bonds single/bond 1 {self.edges[i, 0]+1} {self.edges[i, 1]+1}  ') #special yes?

        #self.lmp.command(f'create_bonds single/bond 1 {self.edges[i, 0] + 1} {self.edges[i, 1] + 1}  ')


        # set callback for helfrich gradient to be handed to lammps vie fix external "ext"

        self.lmp.set_fix_external_callback("ext", self.callback_one, self.lmp)
        #self.L.command(f'create_bonds many all all 1 0.0 100.0 ')




        v=self.lmp.numpy.extract_atom("v")
        self.energy=self.estore.energy(self.mesh.trimesh)+0.5 * v.ravel().dot(v.ravel())
        self.energy_new=0.0
        #self.v=np.zeros()




        #setting and getting helffrich energy from/to lammps
        #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))
        #print(self.lmp.numpy.extract_fix("ext", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, nrow=0))





        # set up flipping

        if self.algo_params.flip_type == "none" or self.algo_params.flip_ratio == 0.0:
            self._flips = lambda: 0
        elif self.algo_params.flip_type == "serial":
            self._flips = lambda: m.flip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        elif self.algo_params.flip_type == "parallel":
            self._flips = lambda: m.pflip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        else:
            raise ValueError("Wrong flip-type: {}".format(self.algo_params.flip_type))

        self.f_i = 0
        self.f_acc = 0

        self.m_i = 0
        self.m_acc = 0
        self.T = self.algo_params.initial_temperature



        # BOOKKEEPING
        self.counter = Counter(move=move_count, flip=flip_count)
        self.timer = Timer(ptime, ptimestamp, timearray, timearray_new,dtimestamp,stime)
        self.output = make_output(self.output_params.output_format, self.output_params.output_prefix,self.output_params.output_counter,callback=self.update_output_counter)
        self.cpt_writer = self.make_checkpoint_handle()
        self.process=psutil.Process()
        self.n = self.algo_params.num_steps // self.output_params.info


        self.info_step = max(self.output_params.info, 0)
        self.out_step = max(self.output_params.thin, 0)
        self.cpt_step = max(self.output_params.checkpoint_every, 0)
        self.refresh_step = max(self.algo_params.refresh, 0)





    #####################  FUNCTIONS FOR FLIPPING


    def lmp_flip_single(self,i,j,k,l):

        self.lmp.command(f'group flip_off id {i} {j}')
        self.lmp.command('delete_bonds flip_off bond 1 remove')
        self.lmp.command('group flip_off clear')
        self.lmp.command(f'create_bonds single/bond 1 {k} {l}')

    def lmp_flip(self,flip_id):

        nf=flip_id[-1][0]

        if nf:

            del_com='remove special'

            for i in range(nf):
                #if i == nf-1:
                del_com = 'remove special'

                self.lmp.command(f'create_bonds single/bond 1 {flip_id[i][0] + 1} {flip_id[i][1] + 1}')
                self.lmp.command(f'group flip_off id {flip_id[i][2] + 1} {flip_id[i][3] + 1}')
                self.lmp.command(f'delete_bonds flip_off bond 1 {del_com}')
                self.lmp.command('group flip_off clear')

                #ids+=f'{flip_id[i*4+2]+1} {flip_id[i*4+3]+1} '
           # print(ids)
           # self.lmp.command(f'group flip_off id {ids}')
           # self.lmp.command('delete_bonds flip_off bond 1 remove special')
           # self.lmp.command('group flip_off clear')
           # for i in range(nf):
           #     self.lmp.command(f'create_bonds single/bond 1 {flip_id[i * 4] + 1} {flip_id[i * 4 + 1] + 1} special yes')
        else:
            pass




    def flip_info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.output_params.info and i_total % self.output_params.info == 0:
            n_edges = self.mesh.trimesh.n_edges()
            ar      = self.f_acc / (self.f_i * n_edges) if not self.f_i == 0 else 0.0
            print("\n-- MCFlips-Step ", self.counter["flip"])
            print("----- flip-accept: ", ar)
            print("----- flip-rate:   ", self.algo_params.flip_ratio)
            self.f_acc = 0
            self.f_i   = 0

    def flip_step(self):
        """Make one step."""
        flip_ids=self._flips()
        self.mesh.f[:]=self.mesh.f
        print(flip_ids)
        self.lmp_flip(flip_ids)


        self.f_acc += flip_ids[-1][0]
        self.f_i += 1
        self.counter["flip"] += 1


    ######################  FUNCTIONS FOR MC



    def hmc_step(self):

        # setting temperature
        i = sum(self.counter.values())
        Tn = np.exp(-self.algo_params.cooling_factor * (i - self.algo_params.start_cooling)) * self.algo_params.initial_temperature
        self.T = max(min(Tn, self.algo_params.initial_temperature), 10e-4)

        self.T=1.0




        # safe mesh for reset in case of rejection
        self.mesh_temp=copy(self.mesh)

        #calute energy
        #future make a flag system to avoid double calculation if lst step was also hmc step

        self.atom_props = f"""     
                        mass            1 1.0
                        velocity        all create {self.T} {np.random.randint(1,9999999)} mom yes dist gaussian   

                        """




        #self.lmp.commands_string(self.atom_props)

        # use ke from lammps to get kinetic energy
        v = self.lmp.numpy.extract_atom("v")


        self.energy = self.estore.energy(self.mesh_temp.trimesh) + 0.5 * v.ravel().dot(v.ravel())
        #self.energy = x[:, 0].dot(x[:, 0])*0.5  + 0.5 * v.ravel().dot(v.ravel())

        #run MD trajectory

        #self.lmp.command(f'run {self.algo_params.traj_steps}')
        #self.mesh.x[:]=self.lmp.numpy.extract_atom("x")

        self.L.run(self.algo_params.traj_steps)

        #set global energy in lammps
        #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))

        # calculate energy difference -> future: do it all in lamps via the set command above (incl. SP and bead interactions)
        v=self.lmp.numpy.extract_atom("v")
        self.mesh.x[:] = self.lmp.numpy.extract_atom("x")
        print(f'{0.5 * v.ravel().dot(v.ravel())} {self.estore.energy(self.mesh.trimesh)}')
        self.energy_new=self.estore.energy(self.mesh.trimesh)+0.5 * v.ravel().dot(v.ravel())


        dh = (self.energy_new- self.energy) / self.T


        # compute acceptance probability: min(1, np.exp(-de))
        a = 1.0 if dh <= 0 else np.exp(-dh)
        u = np.random.uniform()
        acc = u <= a
        if acc:
            self.m_acc += 1

        else:
            # reset positions if rejected
            self.mesh.x[:]=self.mesh_temp.x[:]
            for i in range(v.shape[0]):
                self.L.atoms[i].position[:]=self.mesh_temp.x[i,:]
                #self.lmp.command(f'set atom {i+1} x {self.mesh_temp.x[i,0]} y {self.mesh_temp.x[i,1]} z {self.mesh_temp.x[i,2]} ')

        self.m_i += 1

        self.counter["move"] += 1



    def hmc_info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())

        if self.output_params.info and i_total % self.output_params.info == 0:
            ar = self.m_acc / self.m_i if not self.m_i == 0 else 0.0
            print("\n-- HMC-Step ", self.counter["move"])
            print("----- acc-rate:   ", ar)
            print("----- temperature:", self.T)
            self.m_acc = 0
            self.m_i = 0



    ######################  COMBINED MOVE

    def step(self):

        """Make one step each with each algorithm."""
        if np.random.choice(2) == 0:
            t_fix = time.time()
            self.hmc_step()
            self.timer.timearray_new[0] += (time.time() - t_fix)

        else:
            t_fix = time.time()
            self.flip_step()
            self.timer.timearray_new[1] += (time.time() - t_fix)



    def run(self,N):

        for i in range(N):
            self.step()
            self.hmc_info()
            self.flip_info()
            self.callback(np.copy(self.mesh.x),self.counter)






    # Decorators for meshupdates when calling force function
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


    def _update_mesh_one(func):
        """Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self,  lmp, ntimestep, nlocal, tag, x,f,  *args, **kwargs):

            self.mesh.x = x.reshape(self.mesh.x.shape)
            self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))
            return func(self, lmp, ntimestep, nlocal, tag, x,f, *args, **kwargs)
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
        return wrap






    @_update_mesh_one
    def callback_one(self, lmp, ntimestep, nlocal, tag, x, f):
        #print(tag)
        #tag_clear=[x-1 for x in tag if x <= self.n_vertices]
        f[:]=-self.estore.gradient(self.mesh.trimesh)
        #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))

    @_update_mesh_one
    def callback_harm(self, lmp, ntimestep, nlocal, tag, x, f):
        f[:,0] = -(x[:,0]-2)
        f[:,1] = np.zeros_like(x[:,1])
        f[:, 2] = np.zeros_like(x[:, 2])

        #print(np.max(f))
        # self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))





    #### for minimization
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
    def callback(self,x, steps):




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
            self.L.command(f'write_data lmp_trj/end.p{i}.some')
            #bonds_lmp = self.lmp.numpy.gather_bonds()[:, 1:3]
            #bonds_lmp = np.unique(bonds_lmp, axis=0)
            #bonds_lmp = (np.sort(bonds_lmp, axis=1))

            #with open('bonds_topo.xyz','a+') as f:
            #    for i in range(bonds_lmp.shape[0])
            #    f.write(f'{i}')







        if self.output_params.checkpoint_every and (i % self.output_params.checkpoint_every == 0):
            self.cpt_writer()

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


    # IM PICKLE RIIIICK!!!

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



    ### checkpoints using pickle
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








    ### minimize using gradient for preconditioning

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



    def extra_callback(self, timearray_loc):
        self.timer.timearray_new=timearray_loc



    def update_energy_manager(self):
        self.estore = m.EnergyManager(self.mesh.trimesh, self.eparams)

    def update_energy_parameters(self):
        self.eparams = self.estore.eparams


    def reset_counter(self,move=0,flip=0):

        self.counter = Counter(move=move, flip=flip)




    def reset_output_counter(self):
        self.output_params.output_counter=0

    def update_output_counter(self,ocn):
        self.output_params.output_counter = ocn

    def update_output(self):
        self.output = make_output(self.output_params.output_format, self.output_params.output_prefix,
                                  self.output_params.output_counter, callback=self.update_output_counter)

