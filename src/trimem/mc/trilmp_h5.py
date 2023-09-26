
import h5py
import re
import numpy as np
import trimesh
import pathlib


class H5TrajectoryWriter:
    #Initialize
    def __init__(self,output_params):

        self.op=output_params
        self.fname = pathlib.Path(output_params.output_prefix).with_suffix(".h5")
        self.fname = self._create_part(self.fname)
        self.fnameh5=self.fname
        self.i_traj=0

        print(self.fname)
        print(self.fnameh5)

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


    ## Initializing structure using LAMMPS data

    def _init_struct(self,lmp,mesh,beads,estore):
        h5file = h5py.File(self.fnameh5, "w")
        struct = h5file.create_group(f"/struct")
        struct.attrs["n_trj"] = 0
        struct.attrs["n_vertices"] = mesh.x.shape[0]
        struct.attrs["n_beads"] =beads.n_beads
        struct.attrs["n_edges"] = beads.n_beads
        struct.attrs["n_types"] = beads.n_types


        struct.create_dataset("/struct/ids",
                          data=lmp.numpy.extract_atom('id'),
                          compression="gzip",
                          compression_opts=4)
        struct.create_dataset("/struct/type",
                              data=lmp.numpy.extract_atom('type'),
                              compression="gzip",
                              compression_opts=4)

        sizes=[estore.eparams.bond_params.lc0/2]
        for k in range(beads.n_types):
            if beads.n_types==1:
                sizes.append(beads.bead_sizes/2)
            else:
                sizes.append(beads.bead_sizes[k]/2)

        struct.create_dataset("/struct/sizes",
                              data=np.asarray(sizes),
                              compression="gzip",
                              compression_opts=4)

        h5file.close()



    def _write_state(self, lmp,mesh,i):
        """Write points and edges to the hdf storage."""

        h5file= h5py.File(self.fnameh5, "a")
        #create traj_point
        h5file['struct'].attrs["n_trj"]+=1
        trj = h5file.create_group(f"/{self.i_traj:.0f}/")

        trj.attrs["step"] = i

        # write points
        trj.create_dataset(f"/{self.i_traj:.0f}/particles",
                              data=lmp.numpy.extract_atom('x'),
                              compression="gzip",
                              compression_opts=4)

        # cells
        trj.create_dataset(f"/{self.i_traj:.0f}/edges",
                              data=np.unique(trimesh.Trimesh(vertices=mesh.x, faces=mesh.f).edges_unique,axis=0),
                              compression="gzip",
                              compression_opts=4)

        self.i_traj += 1
        h5file.close()


