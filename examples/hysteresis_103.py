

import sys
from pathlib import Path
sys.path.insert(1,str(Path.home()/'SBEADY_VERSIONS/0.3.1_validation/trimem_sbeady/src'))

from trimem.mc.trisim import TriSim, master_process, read_checkpoint
import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle



mesh = trimesh.creation.icosphere(5)
mesh.vertices=mesh.vertices

print(mesh.vertices.shape[0])
exit()

target_fraction=0.65
initial_fraction=0.3
name='test_com'

increment=0.05
step_sign=np.sign(target_fraction-initial_fraction)
vols=np.arange(initial_fraction,target_fraction+step_sign*increment,step_sign*increment)

do_volume_hysteresis=False


if do_volume_hysteresis:
    with open('benergies.dat', 'w') as f:
        pass

    trisim=TriSim(mesh_points=mesh.vertices,mesh_faces=mesh.faces,
                  initialize=True,output_prefix=name,
                  checkpoint_every=10000,maxiter=200000,
                  out_every=100,thin=2500,num_steps=1100000,start_cooling=500000,
                  info=2500,performance_increment=1000,energy_increment=1000)

    trisim.estore.eparams.volume_frac=0.6
    trisim.update_energy_parameters()
    trisim.minim()
    trisim.estore.eparams.volume_frac = 0.4
    trisim.update_energy_parameters()
    trisim.minim()
#trisim.estore.eparams.curvature_frac = 1.2
  #  trisim.update_energy_parameters()
   # trisim.minim()
    #trisim.estore.eparams.volume_frac = 0.3
    #trisim.update_energy_parameters()
    #trisim.minim()
    trisim.estore.eparams.volume_frac = initial_fraction
    trisim.update_energy_parameters()
    trisim.minim()
    trisim.make_checkpoint(force_name='vol_min.cpt')
    for vol_frac in vols:
        trisim.estore.eparams.volume_frac=vol_frac
        trisim.update_energy_parameters()
        trisim.run()
        trisim.make_checkpoint(force_name=f'vol_{vol_frac:01.3f}.cpt')
        with open('benergies.dat', 'a') as f:
            f.write(f'{vol_frac} {trisim.estore.properties(TriMesh(trisim.mesh.x, trisim.mesh.f)).bending/(8.0*np.pi)}\n')
        trisim.reset_counter()
        #trisim.estore.eparams.continuation_params.lam=0.1

do_test=True
name='trisim_3'
if do_test:
    trisim = TriSim(mesh_points=mesh.vertices, mesh_faces=mesh.faces,
                    initialize=True, output_prefix=name,
                    checkpoint_every=10000, maxiter=30000,
                    out_every=100, thin=25, num_steps=1000, start_cooling=190000, performance_increment=10,
                     info=10,refresh=1000)
    pi=pickle.dumps(trisim)
    trisim=pickle.loads(pi)
    print(trisim.estore.eparams.bond_params.lc1)
   # trisim.estore.eparams.volume_frac = 0.6
   # trisim.update_energy_parameters()
   # trisim.minim()
   # trisim.estore.eparams.volume_frac = 0.4
   # trisim.update_energy_parameters()
   # trisim.minim()
   # trisim.estore.eparams.volume_frac = 0.3
   # trisim.update_energy_parameters()
   # trisim.minim()

    #trisim.make_checkpoint(force_name='vol_min.cpt')
    #trisim.reset_counter()


   # trisim.make_checkpoint(force_name=f'vol_{vol_frac:01.3f}.cpt')
   # for i in range(3):
    trisim.run()
        #trisim.reset_counter()

    print(trisim.estore.gradient(trisim.mesh.trimesh))
    print(trisim.estore.energy(trisim.mesh.trimesh))



do_area_differences=False
target_volume = 1.0
target_curvature = 1.0
name_ad=f'vol{target_volume*100:03.0f}cur{target_curvature*100:03.0f}'


if do_area_differences:
    #with open('benergies.dat', 'w') as f:
    #    pass

    trisim=TriSim(mesh_points=mesh.vertices,mesh_faces=mesh.faces,
                  initialize=True,output_prefix=name_ad,
                  checkpoint_every=10000,maxiter=200000,
                  out_every=100,thin=1000,num_steps=15000000,start_cooling=14000000,
                  reinitialize_every=10000,info=2500)


    trisim.estore.eparams.volume_frac = target_volume
    trisim.update_energy_parameters()
    trisim.minim()
    trisim.estore.eparams.curvature_frac = target_curvature
    trisim.update_energy_parameters()
    trisim.minim()
    trisim.make_checkpoint(force_name=f'{name_ad}_min.cpt')

    trisim.run()
    trisim.make_checkpoint(force_name=f'final_{target_volume*100:03.0f}_{target_curvature*100:03.0f}.cpt')
    with open(f'benergies.dat', 'a+') as f:
        f.write(f'{target_curvature} {trisim.estore.properties(TriMesh(trisim.mesh.x, trisim.mesh.f)).bending/(8.0*np.pi)}\n')
    trisim.reset_counter()

do_maxwell_sampling=False
if do_maxwell_sampling:
    init_temp = 1.0
    name=f'maxwell_T_{init_temp:03.0f}'
    trisim = TriSim(mesh_points=mesh.vertices, mesh_faces=mesh.faces,
                    initialize=True, output_prefix=name,
                    checkpoint_every=100000, maxiter=200000,
                    out_every=1000, thin=5000, num_steps=5000000, start_cooling=5000000,
                    info=2500, performance_increment=1000, energy_increment=50, initial_temperature=init_temp)
    trisim.estore.eparams.volume_frac = 0.95
    trisim.update_energy_parameters()
    trisim.minim()
    trisim.run()


