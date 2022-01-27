"""Checkpoint writer/reader.

Writes/reads the mesh and potentially additional data that is necessary to
run restarts.
"""

import pathlib
import io
import json
import configparser

import numpy as np

from xml.etree import ElementTree as ET
import h5py

from ._common import _create_part

_restart_data = [
    "iteration"
]

class CheckpointWriter:
    """Write checkpoint data to restart mc_app."""

    def __init__(self, fname):
        """Init."""
        self.fname   = pathlib.Path(fname).with_suffix(".cpt")
        self.fnameh5 = self.fname.with_suffix(".cpt.h5")

        # create parts
        self.fname   = _create_part(self.fname)
        self.fnameh5 = _create_part(self.fnameh5)

        self.cpt = ET.Element("cpt", Version="0.1")

    def _write_mesh(self, points, cells):
        """Write points and cells to the hdf storage."""

        h5file = h5py.File(self.fnameh5, "w")

        # write points
        h5file.create_dataset("points",
                              data=points,
                              compression="gzip",
                              compression_opts=4)

        # cells
        h5file.create_dataset("cells",
                              data=cells,
                              compression="gzip",
                              compression_opts=4)

        h5file.close()

    def write(self, points, cells, config, **kwargs):
        """Write points, cells and other data to checkpoint file."""

        # write points information
        xpoints = ET.SubElement(self.cpt,
                                "points",
                                shape="{} {}".format(*points.shape),
                                dtype=points.dtype.name)
        xpoints.text = self.fnameh5.name + ":/points"

        # write cells information
        xcells = ET.SubElement(self.cpt,
                               "cells",
                               shape="{} {}".format(*cells.shape),
                               dtype=cells.dtype.name)
        xcells.text = self.fnameh5.name + ":/cells"

        # write points/cells data
        self._write_mesh(points, cells)

        # write config
        with io.StringIO() as fp:
            config.write(fp)
            config_str = fp.getvalue()
        xconfig = ET.SubElement(self.cpt, "config")
        xconfig.text = json.dumps(config_str)

        # write tree to file
        tree = ET.ElementTree(self.cpt)
        tree.write(self.fname)


class CheckpointReader:
    """Read checkpoint data."""

    def __init__(self, fname, fnum):
        """Init."""
        self.fname   = pathlib.Path(fname).with_suffix(f".p{fnum}.cpt")

        self.tree = ET.parse(self.fname)
        self.root = self.tree.getroot()

    def read(self):
        """Write points, cells and other data to checkpoint file."""

        # read points information
        xpoints = self.root[0]
        if not xpoints.tag == "points":
            raise ValueError("Checkpoint file corrupt.")

        shape   = tuple([int(i) for i in xpoints.attrib["shape"].split()])
        dtype   = xpoints.attrib["dtype"]
        fname   = xpoints.text.split(":/")[0]
        dataset = xpoints.text.split(":/")[-1]

        # read points
        h5file = h5py.File(self.fname.with_name(fname), "r")
        points = np.zeros(shape, dtype=dtype)
        h5file[dataset].read_direct(points)
        h5file.close()

        # read cells information
        xcells = self.root[1]
        if not xcells.tag == "cells":
            raise ValueError("Checkpoint file corrupt.")

        shape   = tuple([int(i) for i in xcells.attrib["shape"].split()])
        dtype   = xcells.attrib["dtype"]
        fname   = xcells.text.split(":/")[0]
        dataset = xcells.text.split(":/")[-1]

        # read cells
        h5file = h5py.File(self.fname.with_name(fname), "r")
        cells   = np.zeros(shape, dtype=dtype)
        h5file[dataset].read_direct(cells)
        h5file.close()

        # read config
        xconfig = self.root[2]
        if not xconfig.tag == "config":
            raise ValueError("Checkpoint file corrupt.")
        config_str = json.loads(xconfig.text)
        config = configparser.ConfigParser()
        config.read_string(config_str)

        return points, cells, config
