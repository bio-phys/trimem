"""A simple xdmf writer.

A simple xdmf writer that writes a series of triangle meshes as
grid collection. It is not a generic xdmf writer but specific to the
use case in trimem, in particular to the writing of temporal series of meshes
as xdmf's collections of grids. Data is stored in hdf5 format.

[https://www.xdmf.org/index.php/XDMF_Model_and_Format]
"""

import pathlib

from ._common import _create_part

from xml.etree import ElementTree as ET
import h5py

class XdmfWriter:
    """Xdmf writer to write series of meshes.

    Args:
        fname (str, path-like): output file prefix.
    """

    def __init__(self, fname):
        """Init."""
        self.fname   = pathlib.Path(fname).with_suffix(".xmf")
        self.fnameh5 = self.fname.with_suffix(".h5")

        # create parts
        self.fname   = _create_part(self.fname)
        self.fnameh5 = _create_part(self.fnameh5)

        self.h5file = h5py.File(self.fnameh5, "w")

        self.data_counter = 0
        self.step_counter = 0

        # init xml tree
        self.xdmf   = ET.Element("Xdmf", Version="3.0")
        self.domain = ET.SubElement(self.xdmf, "Domain")
        self.grids  = ET.SubElement(self.domain,
                                    "Grid",
                                    Name="Trajectory",
                                    GridType="Collection",
                                    CollectionType="Temporal")

        # init time steps
        self.time     = ET.SubElement(self.grids, "Time", TimeType="HyperSlab")
        self.timedata = ET.SubElement(self.time,
                                      "DataItem",
                                      NumberType="Int",
                                      Dimensions="3")

    def _write_time(self):
        """Write hyperslab 'start stride count' string."""
        return "0 1 {}".format(self.step_counter)

    def _write_data(self, data):
        """Write data to the hdf storage."""
        name = "data{}".format(self.data_counter)
        self.h5file.create_dataset(name,
                                   data=data,
                                   compression="gzip",
                                   compression_opts=4)

        self.data_counter += 1

        # give file name (relative to xdmf container file!)
        return self.fnameh5.name + ":/" + name

    def write_points_cells(self, points, cells):
        """Write points and cells by appending a grid to the collection.

        Args:
            points (ndarray[float]): (N,3) array of vertex positions with N
                being the number of vertices.
            cells (ndarray[int]): (M,3) array of face definitions with M
                being the number of faces.
        """

        step = "Step {}".format(self.step_counter)

        grid = ET.SubElement(self.grids, "Grid", Name=step, GridType="Uniform")

        # Geometry
        geom = ET.SubElement(grid, "Geometry", GeometryType="XYZ")

        dtype = points.dtype.name.rstrip("123468").capitalize()
        prec  = str(points.dtype.itemsize)
        dim   = "{} {}".format(*points.shape)
        data  = ET.SubElement(geom,
                              "DataItem",
                              DataType=dtype,
                              Dimensions=dim,
                              Format="HDF",
                              Precision=prec)
        data.text = self._write_data(points)

        # Topology
        topo = ET.SubElement(grid,
                             "Topology",
                             TopologyType="Triangle",
                             NumberOfElements=str(cells.shape[0]),
                             NodesPerElement=str(cells.shape[1]))
        dtype = cells.dtype.name.rstrip("123468").capitalize()
        prec  = str(cells.dtype.itemsize)
        dim   = "{} {}".format(*cells.shape)
        data  = ET.SubElement(topo,
                              "DataItem",
                              DataType=dtype,
                              Dimensions=dim,
                              Format="HDF",
                              Precision=prec,
        )
        data.text = self._write_data(cells)

        # update time steps
        self.timedata.text = self._write_time()

        self.step_counter +=1

        tree = ET.ElementTree(self.xdmf)
        tree.write(self.fname)
