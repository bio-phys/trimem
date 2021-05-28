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
        (0.0, 0.0),
        (l, l),
        n=n,  # or (11, 11)
        variant="zigzag",  # or "up", "down", "center"
    )
    x = np.append(points, np.zeros((len(points),1)), axis=1)
    return om.TriMesh(x, cells)

# -----------------------------------------------------------------------------
#                                                                      checks -
# -----------------------------------------------------------------------------
def valid_cell_list():
    """Visualize cells."""
    mesh = get_mesh(20)
    x = mesh.points()

    c = m.CellList(mesh, 0.2)
    print("Cells::shape:", c.shape)
    print("Cells::stride:", c.strides)
    print("Cells::r_list:", c.r_list)
    print("Cells::pairs:", c.cell_pairs)
    print("len(Cells::pairs):", len(c.cell_pairs))

    for vi in c.cells.values():
        plt.plot(x[vi,0], x[vi,1], '.')

    # check distance counts
    d = c.distance_counts(mesh, 0.2)
    print("num pairs in r:",d)

    # compare agains kd-tree distance counts
    tree = KDTree(x)
    d = tree.count_neighbors(tree, 0.2)
    print("num pairs in r from tree:", d)

    # check distance computation
    d,i,j = c.distance_matrix(mesh, 0.2)
    A = coo_matrix((d,(i,j)))
    B = coo_matrix((d,(j,i)))
    M = A + B

    plt.matshow(M.toarray())
    plt.show()

    # compare against kd-tree distance computation
    C = tree.sparse_distance_matrix(tree, 0.2)
    plt.matshow(M.toarray()-C.toarray())
    plt.show()


def valid_nlist():
    """Check nlists."""
    mesh = get_mesh(20)
    x = mesh.points()

    n = m.rNeighbourList(mesh, 0.2)

    # plot some neighbourhood
    idx = 50
    plt.plot(x[:,0], x[:,1], '.')
    plt.plot(x[idx,0], x[idx,1], 'o', color='b')
    p = n.neighbours[idx]
    plt.plot(x[p,0], x[p,1], 'o', color='orange', alpha=0.3)
    plt.show()

    # check distance computation
    d,i,j = n.distance_matrix(mesh, 0.2)
    A = coo_matrix((d,(i,j)))

    plt.matshow(A.toarray())
    plt.show()

    # validate against direct cell based computation
    c = m.CellList(mesh, 0.2)
    d,i,j = c.r_distance_matrix(mesh, 0.2)
    B = coo_matrix((d,(i,j)), shape=(len(x),len(x)))
    B = B+B.transpose()

    plt.matshow(A.toarray()-B.toarray())
    plt.show()

def check_scaling():
    """Check linear order of distance computation."""

    dts_trimem = []
    dts_kdtree = []
    dims = [2**3, 2**4, 2**5, 2**6, 2**7, 2**8]
    N    = [d**2 for d in dims] # mesh gen takes sqrt(n) as input
    for n in dims:
        mesh = get_mesh(n, constant_density=True)
        points = mesh.points()

        start = time()
        nl = m.CellList(mesh, 0.2)
#        nl = m.NeighbourList(mesh, 0.2)
        d = nl.distance_counts(mesh, 0.2)
#        d,i,j = m.distance_matrix(mesh, nl, 0.2)
        dts_trimem.append(time()-start)

        start = time()
        tree = KDTree(points)
        d = tree.count_neighbors(tree, 0.2)
        print(d)
#        d = tree.sparse_distance_matrix(tree, 0.2, output_type="coo_matrix")
        dts_kdtree.append(time()-start)

        print("Compute distances for {} vertices took:".format(n**2))
        print("  trimem {}s".format(dts_trimem[-1]))
        print("  kdtree {}s".format(dts_kdtree[-1]))

    # references
    cli = dts_trimem[0]/N[0]
    csq = dts_trimem[0]/N[0]**2
    squ = [csq*x**2 for x in N]
    lin = [cli*x for x in N]

#    plt.plot(N, squ, "--", label=r'x^2', alpha=0.5)
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

def test_3d():
    """Test in 3d."""
    points, cells = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(points, cells)

    c = m.rNeighbourList(mesh, 0.2)

    start = time()
    d = c.distance_counts(mesh, 0.01)
    dt = time()-start
    print(d)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = points
    plt.plot(x[:,0], x[:,1], x[:,2], '.')
    plt.plot(x[25,0], x[25,1], x[25,2], 'o', color='b')
    print(c.neighbours[25])
    for p in c.neighbours[25]:
        plt.plot(x[p,0], x[p,1], x[p,2], 'o', color='orange', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    #valid_cell_list()
    #valid_nlist()

    check_scaling()

    #test_3d()
