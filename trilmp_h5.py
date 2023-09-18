
import h5py
import re
import numpy as np
import trimesh


class H5TrajectoryWriter():

    #Initialize
    def __init__(self,trilmp):
        self.op=trilmp.output_params
        self.fnameh5=f'{self._get_parts(self.op.output_prefix)}.h5'
        self.i_traj=0

    # NAMING SHEME STOLEN FROM TRIMEM -> checks wheter a output_suffix.p*.h5 exists or not. If yes creates file with *+1 to be used to write
    # trajectory to
    def _remsuffix(self,string, suffix):
        """removesuffix for py <3.9 and without checks. Internal use only."""
        return string[:-len(suffix)]

    def _get_pattern(self,fname):
        """Create part enumeration pattern '{fname.stem}.p{}.{fname.suffix}'."""
        suffix = fname.suffix
        stem = fname.stem
        return f"{stem}.p{{}}{suffix}"

    def _get_parts(self,fname):
        """Get number of exisiting parts with enumeration pattern (_get_pattern).

        This routine currently just counts exisiting files with the pattern
        and returns the count. Potential gaps in file enumeration are thus not
        recognized nor supported currently.
        """

        pattern = self._get_pattern(fname)

        # check how many parts already exist
        parts = []
        for f in fname.parent.iterdir():
            repat = pattern.replace(".", "\.")
            m = re.match(repat.format("[0-9]+"), f.name)
            if not m is None:
                num_str = self._remsuffix(m.string, fname.suffix).split(".")[-1][1:]
                parts.append(int(num_str))

        return len(parts)

    def _create_part(self,fname):
        """Create enumerated file-part."""

        pattern = self._get_pattern(fname)

        new_part = self._get_parts(fname)

        # rename with new part-number
        return fname.with_name(pattern.format(new_part))

    def _init_struct(self,trilmp):
        h5file = h5py.File(self.fnameh5, "w")
        struct = h5file.create_group(f"/struct")
        struct.attrs["n_vertices"] = trilmp.n_vertices
        struct.attrs["n_beads"] =trilmp.beads.n_beads
        struct.attrs["n_edges"] = trilmp.beads.n_beads
        struct.attrs["n_types"] = trilmp.beads.n_types


        struct.create_dataset(f"/struct/ids",
                          data=trilmp.lmp.numpy.extract_atom('id'),
                          compression="gzip",
                          compression_opts=4)


        struct.create_dataset(f"/struct/ids",
                          data=trilmp.lmp.numpy.extract_atom('id'),
                          compression="gzip",
                          compression_opts=4)

        h5file.close()



    def _write_state(self, trilmp):
        """Write points and cells to the hdf storage."""

        np.sum(trilmp.counter.values())

        h5file= h5py.File(self.fnameh5, "a")
        #create traj_point
        trj = h5file.create_group(f"/{self.i_traj:.0f}/")
        trj.attrs["step"] =np.sum()

        # write points
        trj.create_dataset(f"/{self.i_traj:.0f}/particles",
                              data=trilmp.lmp.numpy.extract_atom('x'),
                              compression="gzip",
                              compression_opts=4)

        # cells
        trj.create_dataset(f"/{self.i_traj:.0f}/edges",
                              data=np.unique(trimesh.Trimesh(vertices=trilmp.mesh.x, faces=trilmp.mesh.f).edges_unique,axis=0),
                              compression="gzip",
                              compression_opts=4)

        if self.op.energy

        h5file.close()


def create_h5_simulation_file(name, mode, eta, eps, lam, Lx, Ly, M, N, rho0, mu, p_bulk, mu_full, rank, i_run=0,
                              write_mode='w'):
    if rank == 0:
        f = h5py.File(f'{name}', write_mode)

        par = f.create_group(f"/{i_run:03.0f}/parameters")
        par.attrs["mode"] = mode
        par.attrs["eta"] = eta
        par.attrs["eps"] = eps
        par.attrs["lam"] = lam
        par.attrs["Lx"] = Lx
        par.attrs["Ly"] = Ly
        par.attrs["M"] = M
        par.attrs["N"] = N

        bul = f.create_group(f"/{i_run:03.0f}/bulk")
        bul.attrs["rho0"] = rho0
        bul.attrs["mu"] = mu
        bul.attrs["p_bulk"] = p_bulk
        bul.attrs["mu_full"] = mu + np.log(rho0)

        rhg = f.create_group(f"/{i_run:03.0f}/rho")
        obg = f.create_group(f"/{i_run:03.0f}/obs")
        obg.create_dataset(f'/{i_run:03.0f}/obs/pressure', data=np.zeros((1, 4)), chunks=True, maxshape=(None, 4))

        f.close()

    return


def write_density_to_h5_simulation_file(name, rho, nstep, comm, comm_size, rank, xshape, i_run=0, omega=0, p=0,
                                        convergence_status='n', err=0):
    rhog = create_global_array(comm, comm_size, rank, rho, xshape, 818)

    if rank == 0:
        file = h5py.File(f'{name}', 'a')

        dset = file[f"/{i_run:03.0f}/rho"].create_dataset(f"/{i_run:03.0f}/rho/{nstep:08.0f}", shape=rhog.shape,
                                                          data=rhog, chunks=True)
        file[f"/{i_run:03.0f}/rho"].attrs['last'] = f'{nstep:08.0f}'
        file[f"/{i_run:03.0f}/rho"].attrs['converged'] = convergence_status
        dset.attrs['omega'] = omega
        dset.attrs['p'] = p
        dset.attrs['err'] = err
        file.close()
    return


def save_upscaled_density_to_h5_simulation_file(name, rhog, nstep, rank, i_run=0, omega=0, p=0,
                                                convergence_status='n', err=0):
    if rank == 0:
        file = h5py.File(f'{name}', 'a')

        dset = file[f"/{i_run:03.0f}/rho"].create_dataset(f"/{i_run:03.0f}/rho/{nstep:08.0f}", shape=rhog.shape,
                                                          data=rhog, chunks=True)
        file[f"/{i_run:03.0f}/rho"].attrs['last'] = f'{nstep:08.0f}'
        file[f"/{i_run:03.0f}/rho"].attrs['converged'] = convergence_status
        dset.attrs['omega'] = omega
        dset.attrs['p'] = p
        dset.attrs['err'] = err
        file.close()
    return


def write_observables_to_h5_simulation_file(name, nstep, omega, p, den, rank, i_run=0):
    if rank == 0:
        file = h5py.File(f'{name}', 'a')
        data = np.zeros((1, 4))
        data[0, 0] = nstep
        data[0, 1] = omega
        data[0, 2] = p
        data[0, 3] = den

        file[f"/{i_run:03.0f}/obs/pressure"].resize((file[f"/{i_run:03.0f}/obs/pressure"].shape[0] + data.shape[0]),
                                                    axis=0)
        file[f"/{i_run:03.0f}/obs/pressure"][-data.shape[0]:] = data
        file.close()
    return


### for specific state nstep has to be input in string format i.e. '001000' for nstep=1000
def load_input_from_h5_simulation_file(name, use_initial_state_from_file, use_parameters_from_file, nstep='last',
                                       i_run=0):
    file = h5py.File(f'{name}', 'r')

    if use_parameters_from_file:
        MODE = file[f'{i_run:03.0f}/parameters'].attrs["mode"]
        eta = file[f'{i_run:03.0f}/parameters'].attrs["eta"]
        eps = file[f'{i_run:03.0f}/parameters'].attrs["eps"]
        lam = file[f'{i_run:03.0f}/parameters'].attrs["lam"]
        Lx = file[f'{i_run:03.0f}/parameters'].attrs["Lx"]
        Ly = file[f'{i_run:03.0f}/parameters'].attrs["Ly"]
        M = file[f'{i_run:03.0f}/parameters'].attrs["M"]
        N = file[f'{i_run:03.0f}/parameters'].attrs["N"]

        rho0 = file[f'/{i_run:03.0f}/bulk'].attrs["rho0"]
        mu = file[f'/{i_run:03.0f}/bulk'].attrs["mu"]
        p_bulk = file[f'/{i_run:03.0f}/bulk'].attrs["p_bulk"]
        mu_full = file[f'/{i_run:03.0f}/bulk'].attrs["mu_full"]

    if use_initial_state_from_file:
        if nstep == 'last':
            nstep = file[f'/{i_run:03.0f}/rho'].attrs["last"]

        rho_global = file[f'/{i_run:03.0f}/rho/{nstep}'][:]

    file.close()

    if use_parameters_from_file and use_initial_state_from_file:
        return rho_global, MODE, eta, eps, lam, Lx, Ly, M, N, rho0, mu, p_bulk, mu_full

    if use_parameters_from_file and not (use_initial_state_from_file):
        return MODE, eta, eps, lam, Lx, Ly, M, N, rho0, mu, p_bulk, mu_full

    if not (use_parameters_from_file) and use_initial_state_from_file:
        return rho_global
