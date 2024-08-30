from ovito.data import DataCollection
from ovito.io import FileReaderInterface, import_file
from typing import Callable, Any
import re
import h5py

class TrilmpH5Reader(FileReaderInterface):

    @staticmethod
    def detect(filename: str):
        try:
            with h5py.File(filename, "r") as f:
                return f["struct"].attrs["n_trj"] != 0
        except OSError:
            return False

    def scan(self, filename: str, register_frame: Callable[..., None]):

        with h5py.File(filename, "r") as f:
            for i in range(f["struct"].attrs["n_trj"]):
                register_frame(frame_info=(i,f["struct"].attrs["n_vertices"]+f["struct"].attrs["n_beads"],f["struct"].attrs["n_edges"] ), label=f["/{self.i_traj:.0f}"].attrs["step"])

    def parse(self, data: DataCollection, filename: str, frame_info: tuple[int,int, int], **kwargs: Any):
        i_frame, num_particles, num_edges = frame_info

        with h5py.File(filename, "r") as f:
            particles = data.create_particles(count=num_particles)
            positions = particles.create_property("Position")
            positions[:,:]=f[f"/{i_frame}/particles"]
            particle_types = particles.create_property("Particle Type")
            particle_types[:,:]=f["struct/type"]
            bonds = data.particles_.create_bonds(count=num_edges)
            bonds.create_property('Topology', data=f[f"/{i_frame}/edges"])
