import numpy as np
import pytest
import h5py

from xml.etree import ElementTree as ET

import meshio

from helfrich.mc.output import make_output

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(scope="function")
def outdir(tmp_path_factory):
    """Output directory."""
    return tmp_path_factory.mktemp("testout")

@pytest.fixture()
def data():
    """Some test points and cells."""
    p = np.random.randn(4,3)
    c = np.array([[0,1,2],[2,0,3]])
    return p, c

@pytest.fixture(params=["xyz", "xdmf", "vtu"])
def writer_type(request):
    return request.param

@pytest.fixture()
def writer(outdir, writer_type):
    prefix = outdir.joinpath("tmp").resolve()
    config = {"GENERAL": {"output_format": writer_type,
                          "output_prefix": str(prefix)}}
    return make_output(config)
    
# -----------------------------------------------------------------------------
#                                                                       test --
# -----------------------------------------------------------------------------
def test_writer(writer, data):
    """Verify output."""

    points, cells = data

    # write two steps (using the same data)
    writer.write_points_cells(points, cells)
    writer.write_points_cells(points, cells)

    # check write 
    assert writer.fname.exists()

    if writer.fname.suffix == ".xyz":
        outfile = writer.fname
        # check content of step 1
        re_data = np.loadtxt(outfile, usecols=[1,2,3], skiprows=1, max_rows=4)
        assert np.linalg.norm(points - re_data) < 1.0e-5

        # check content of step 2
        re_data = np.loadtxt(outfile, usecols=[1,2,3], skiprows=7, max_rows=4)
        assert np.linalg.norm(points - re_data) < 1.0e-5

    elif writer.fname.suffix == ".vtu":
        # check content of step 1
        fname = writer.fname.parent.joinpath("tmp.0.p0.vtu")
        mesh = meshio.read(fname)
        assert np.linalg.norm(points - mesh.points) < 1.0e-8
        assert np.linalg.norm(cells - mesh.cells[0].data) == 0.0

        # check content of step 2
        fname = writer.fname.parent.joinpath("tmp.1.p0.vtu")
        mesh = meshio.read(fname)
        assert np.linalg.norm(points - mesh.points) < 1.0e-8
        assert np.linalg.norm(cells - mesh.cells[0].data) == 0.0

    elif writer.fname.suffix == ".xmf":
        # get h5file
        h5filename  = writer.fname.with_suffix(".h5")

        # check write 
        assert h5filename.exists()

        re_points = np.zeros_like(points)
        re_cells  = np.zeros_like(cells)
        h5file = h5py.File(h5filename, "r")
        
        tree = ET.parse(writer.fname)
        root = tree.getroot()

        assert root.tag == "Xdmf"

        domain = root[0]
        assert domain.tag == "Domain"

        collection = domain[0]
        assert collection.tag == "Grid"
        assert collection.attrib["CollectionType"] == "Temporal"
        assert collection.attrib["GridType"] == "Collection"
        assert collection.attrib["Name"] == "Trajectory"
        assert len(collection) == 2

        # first grid
        grid1 = collection[0]
        assert grid1.tag == "Grid"
        assert grid1.attrib["Name"] == "Step 0"
        assert grid1.attrib["GridType"] == "Uniform"
        assert len(grid1) == 2

        geom = grid1[0]
        assert geom.tag == "Geometry"
        assert geom.attrib["GeometryType"] == "XYZ"

        data = geom[0]
        assert data.tag == "DataItem"
        assert data.attrib["DataType"] == "Float"
        assert data.attrib["Dimensions"] == "{} {}".format(*points.shape)
        assert data.attrib["Format"] == "HDF"
        assert data.text.split(":")[0] == h5filename.name
        assert data.text.split(":")[1] == "/data0"
        h5file["data0"].read_direct(re_points)
        assert np.linalg.norm(points - re_points) < 1.0e-8

        top = grid1[1]
        assert top.tag == "Topology"
        assert top.attrib["NodesPerElement"] == "3"
        assert top.attrib["NumberOfElements"] == "{}".format(len(cells))
        assert top.attrib["TopologyType"] == "Triangle"

        data = top[0]
        assert data.tag == "DataItem"
        assert data.attrib["DataType"] == "Int"
        assert data.attrib["Dimensions"] == "{} {}".format(*cells.shape)
        assert data.attrib["Format"] == "HDF"
        assert data.text.split(":")[0] == h5filename.name
        assert data.text.split(":")[1] == "/data1"
        h5file["data1"].read_direct(re_cells)
        assert np.linalg.norm(cells - re_cells) < 1.0e-12

        # second grid
        grid1 = collection[1]
        assert grid1.tag == "Grid"
        assert grid1.attrib["Name"] == "Step 1"
        assert grid1.attrib["GridType"] == "Uniform"
        assert len(grid1) == 2

        geom = grid1[0]
        assert geom.tag == "Geometry"
        assert geom.attrib["GeometryType"] == "XYZ"

        data = geom[0]
        assert data.tag == "DataItem"
        assert data.attrib["DataType"] == "Float"
        assert data.attrib["Dimensions"] == "{} {}".format(*points.shape)
        assert data.attrib["Format"] == "HDF"
        assert data.text.split(":")[0] == h5filename.name
        assert data.text.split(":")[1] == "/data2"
        h5file["data2"].read_direct(re_points)
        assert np.linalg.norm(points - re_points) < 1.0e-8

        top = grid1[1]
        assert top.tag == "Topology"
        assert top.attrib["NodesPerElement"] == "3"
        assert top.attrib["NumberOfElements"] == "{}".format(len(cells))
        assert top.attrib["TopologyType"] == "Triangle"

        data = top[0]
        assert data.tag == "DataItem"
        assert data.attrib["DataType"] == "Int"
        assert data.attrib["Dimensions"] == "{} {}".format(*cells.shape)
        assert data.attrib["Format"] == "HDF"
        assert data.text.split(":")[0] == h5filename.name
        assert data.text.split(":")[1] == "/data3"
        h5file["data3"].read_direct(re_cells)
        assert np.linalg.norm(cells - re_cells) < 1.0e-12

def test_output_versioning(outdir, data, writer_type):
    """Test versioning of output."""

    prefix = outdir.joinpath("tmp").resolve()
    config = {"GENERAL": {"output_format": writer_type,
                          "output_prefix": str(prefix)}}

    # first version
    writer = make_output(config)
    writer.write_points_cells(data[0], data[1])

    if writer.fname.suffix == ".xyz":
        assert prefix.with_suffix(".p0.xyz").exists()
    elif writer.fname.suffix == ".xmf":
        assert prefix.with_suffix(".p0.xmf").exists()
        assert prefix.with_suffix(".p0.h5").exists()
    elif writer.fname.suffix == ".vtu":
        assert prefix.with_suffix(".0.p0.vtu").exists()

    # second version
    writer = make_output(config)
    writer.write_points_cells(data[0], data[1])

    if writer.fname.suffix == ".xyz":
        assert prefix.with_suffix(".p1.xyz").exists()
    elif writer.fname.suffix == ".xmf":
        assert prefix.with_suffix(".p1.xmf").exists()
        assert prefix.with_suffix(".p1.h5").exists()
    elif writer.fname.suffix == ".vtu":
        assert prefix.with_suffix(".0.p1.vtu").exists()
