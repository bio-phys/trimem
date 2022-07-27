"""Utilities for the tests."""
import numpy as np
import trimesh

def tube(length=1.0, radius=1.0, segments=30):
    """A tubular mesh around the y-axis."""

    U = 2 * np.pi * radius # circumcircle
    t = U / segments # mesh size
    n = int(length//t) + 1 # num elements along length

    # angles of revolution around y-axes
    angles = np.linspace(0, 2*np.pi, segments+1)[:-1]

    # coordinates
    y = np.linspace(-length/2, length/2, n)
    x = np.cos(angles)*radius
    z = np.sin(angles)*radius

    verts = np.hstack(
                (
                    np.tile(x, (1,n)).T,
                    np.repeat(y, segments).reshape((segments * n,1)),
                    np.tile(z, (1,n)).T,
                )
            )

    idx = np.arange(segments)

    # faces of one ring of triangles
    faces = np.vstack((
        np.array([
            idx, idx+segments, (idx+1)%segments
        ]).T,
        np.array([
            (idx+1)%segments, idx+segments, (idx+1)%segments+segments
        ]).T
    ))

    # repeat for n times
    faces = np.vstack([faces+(i*segments) for i in range(n-1)])

    return verts, faces


def rect(nx,ny):
    """A rectangular mesh."""

    xx, yy = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))

    verts = np.hstack((xx.reshape((nx*ny,1)), yy.reshape((nx*ny,1)), np.zeros((nx*ny,1))))

    ii = np.arange(nx)

    # faces of first row
    faces = (
        np.array([
            ii[:nx-1], ii[1:nx], ii[1:nx]+nx
        ]).T,
        np.array([
            ii[:nx-1]+nx, ii[:nx-1], ii[1:nx]+nx
        ]).T,
    )
    faces = np.vstack(faces)

    # repeat ny-1 times
    faces = np.vstack([faces+(i*nx) for i in range(ny-1)])

    return verts, faces

def icosphere(n):
    """Icosphere from trimesh."""
    s = trimesh.creation.icosphere(n)
    return s.vertices, s.faces

def icosahedron():
    """icosahedron from trimesh."""
    s = trimesh.creation.icosahedron()
    return s.vertices, s.faces
