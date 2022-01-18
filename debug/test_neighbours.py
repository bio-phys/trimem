import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree
from time import time

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#                                                                       utils -
# -----------------------------------------------------------------------------
def get_mesh(n, constant_density=False):

    l = 1.0

    if constant_density:
        l = l/20*n

    points, cells = meshzoo.rectangle_tri(
        np.linspace(0.0, l, n),
        np.linspace(0.0, l, n),
        variant="zigzag",  # or "up", "down", "center"
    )
    x = np.append(points, np.zeros((len(points),1)), axis=1)
    return om.TriMesh(x, cells)

def get_nlist(mesh, ltype, rlist, excl):
    """Get a neighbour list."""

    params = m.EnergyParams()
    params.repulse_params.n_search        = ltype
    params.repulse_params.rlist           = rlist
    params.repulse_params.exclusion_level = excl

    return m.make_nlist(mesh, params)

# -----------------------------------------------------------------------------
#                                                                      checks -
# -----------------------------------------------------------------------------
def valid_lists(ltype="cell-list"):
    """Validate neighbour lists."""
    mesh = get_mesh(20)
    x = mesh.points()

    nl = get_nlist(mesh, ltype, 0.2, 0)

    # compute distance matrix
    d,i,j = nl.distance_matrix(mesh, 0.123)
    A = coo_matrix((d,(i,j)), shape=(len(x),len(x)))
    M = A + A.T # kdtree gives full matrix

    # compare against kd-tree distance computation
    tree = KDTree(x)
    C = tree.sparse_distance_matrix(tree, 0.123)

    assert (C-M).max() == 0.0

def vis_neighbourhood(ltype="cell-list"):
    """Check nlists."""
    mesh = get_mesh(20)
    x = mesh.points()

    nl = get_nlist(mesh, ltype, 0.2, 0)

    # plot some neighbourhood
    idx = 151
    _,jdx = nl.point_distances(mesh, idx, 0.123)
    plt.plot(x[:,0], x[:,1], '.', color='k', alpha=0.3)
    plt.plot(x[idx,0], x[idx,1], 'o', color='b', alpha=0.7)
    plt.plot(x[jdx,0], x[jdx,1], 'o', color='orange', alpha=0.5)
    plt.title("{} neighbours".format(len(jdx)))
    plt.show()

def check_scaling(ltype="cell-list", dimp2=8):
    """Check linear order of distance computation."""

    dts_trimem = []
    dts_kdtree = []
    dims = [2**p2 for p2 in range(3,dimp2+1)]
    N    = [d**2 for d in dims] # mesh gen takes sqrt(n) as input
    for n in dims:
        mesh = get_mesh(n, constant_density=True)
        points = mesh.points()

        start = time()
        nl = get_nlist(mesh, ltype, 0.2, 0)
        secl = time()-start
        for i in range(10):
            d,i,j = nl.distance_matrix(mesh, 0.2)
        dts_trimem.append(time()-start)

        start = time()
        tree = KDTree(points)
        setr = time()-start
        for i in range(10):
            d = tree.sparse_distance_matrix(tree, 0.2, output_type="coo_matrix")
        dts_kdtree.append(time()-start)

        print("Compute distances for {} vertices took:".format(n**2))
        print("  trimem {}s".format(dts_trimem[-1]))
        print("  kdtree {}s".format(dts_kdtree[-1]))
        print("  setup cost clist(%) :", secl/dts_trimem[-1])
        print("  setup cost kdtree(%):", setr/dts_kdtree[-1])

    # references
    cli = dts_trimem[0]/N[0]
    lin = [cli*x for x in N]

    plt.plot(N, dts_trimem, "o-", label="trimem")
    plt.plot(N, dts_kdtree, "o-", label="kdtree")
    plt.plot(N, lin, "--", label=r'O(N)', alpha=0.5, color="gray")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Scaling of r-ball search")
    plt.xlabel("number of vertices")
    plt.ylabel("time to solution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    valid_lists(ltype="cell-list")
    valid_lists(ltype="verlet-list")

    vis_neighbourhood(ltype="cell-list")
    vis_neighbourhood(ltype="verlet-list")

    check_scaling(ltype="cell-list")
    check_scaling(ltype="verlet-list")
