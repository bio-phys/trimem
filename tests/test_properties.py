import trimem.core as m
import autograd.numpy as np
from autograd import grad

import pytest

# -----------------------------------------------------------------------------
#                                                                       util --
# -----------------------------------------------------------------------------
def edge_vector(points):
    return points[1] - points[0]

def edge_length(points):
    return np.linalg.norm(edge_vector(points))

def face_normal_u(points):
    v2 = points[2] - points[1]
    v0 = points[0] - points[1]
    return np.cross(v0,v2)

def face_area(points):
    n = face_normal_u(points)
    print(n)
    return np.linalg.norm(n) / 2

def face_volume(points):
    return np.dot(points[0], np.cross(points[1],points[2])) / 6

def dihedral_angle(points):
    n0 = face_normal_u(points[[0,1,2]])
    n1 = face_normal_u(points[[1,0,3]])
    n0 /= np.linalg.norm(n0)
    n1 /= np.linalg.norm(n1)

    ve = edge_vector(points[[0,1]])
    sgn = np.dot(np.cross(n0,n1),ve)
    alpha = np.arccos(np.dot(n0,n1))
    return alpha if sgn > 0 else -alpha

d_edge_length    = grad(edge_length)
d_face_area      = grad(face_area)
d_face_volume    = grad(face_volume)
d_dihedral_angle = grad(dihedral_angle)

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(params=[np.pi/4, -np.pi/4])
def data(request):
    """facet-pair with properties for testing."""

    alpha = request.param
    x = np.cos(alpha) * 0.5
    z = np.sin(alpha) * 0.5

    points = np.array([[0.0, -0.5, 0.0],
                       [0.0,  0.5, 0.0],
                       [ -x,  0.0,   z],
                       [  x,  0.0,   z]])
    cells = np.array([[0,1,2],[1,0,3]])

    results = {}
    results["length"]   = edge_length(points[[0,1]])
    results["d_length"] = d_edge_length(points[[0,1]])

    results["area"]     = face_area(points[[0,1,2]])
    results["d_area"]   = d_face_area(points[[0,1,2]])

    results["volume"]   = face_volume(points[[0,1,2]])
    results["d_volume"] = d_face_volume(points[[0,1,2]])

    results["angle"]    = dihedral_angle(points)
    results["d_angle"]  = d_dihedral_angle(points)

    mesh = m.TriMesh(points, cells)

    class Data:
        pass

    d = Data()
    d.mesh = mesh
    d.ref  = results

    yield d

# -----------------------------------------------------------------------------
#                                                                      tests --
# -----------------------------------------------------------------------------
def test_edge_length(data):
    """test edge_length."""

    mesh = data.mesh
    ref  = data.ref
    e = mesh.halfedge_handle(0)

    l  = m.edge_length(mesh, e)
    dl = m.edge_length_grad(mesh, e)

    # reference results
    r_l  = ref["length"]
    r_dl = ref["d_length"]

    assert np.abs(l - r_l)/r_l < 1.0e-10
    assert np.linalg.norm(dl - r_dl)/np.linalg.norm(r_dl) < 1.0e-10

def test_area(data):
    """test face area."""

    mesh = data.mesh
    ref  = data.ref
    e = mesh.halfedge_handle(0)

    a  = m.face_area(mesh, e)
    da = m.face_area_grad(mesh, e)

    # reference results
    r_a  = ref["area"]
    r_da = ref["d_area"]

    assert np.abs(a - r_a)/r_a < 1.0e-10
    assert np.linalg.norm(da - r_da)/np.linalg.norm(r_da) < 1.0e-10

def test_volume(data):
    """test face volume."""

    mesh = data.mesh
    ref  = data.ref
    e = mesh.halfedge_handle(0)

    v  = m.face_volume(mesh, e)
    dv = m.face_volume_grad(mesh, e)

    # reference results
    r_v  = ref["volume"]
    r_dv = ref["d_volume"]

    assert np.abs(v - r_v) < 1.0e-10
    assert np.linalg.norm(dv - r_dv)/np.linalg.norm(r_dv) < 1.0e-10

def test_angle(data):
    """test dihedral_angle."""

    mesh = data.mesh
    ref  = data.ref
    e = mesh.halfedge_handle(0)

    a  = m.dihedral_angle(mesh, e)
    da = m.dihedral_angle_grad(mesh, e)

    # reference results
    r_a  = ref["angle"]
    r_da = ref["d_angle"]

    assert np.abs(a - r_a) < 1.0e-10
    assert np.linalg.norm(da - r_da)/np.linalg.norm(r_da) < 1.0e-10
