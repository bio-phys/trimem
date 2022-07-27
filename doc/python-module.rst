.. _python-module:

Python interface
================

The :ref:`cli` provides high-level usage of trimem offering access to
minimization and Monte Carlo sampling of the Helfrich functional
parametrized by a :ref:`config-file`. Checkpointing of the simulations thereby
enables a very versatile chaining of simulation steps.

The python interface presented here enables access to the evaluation of the
Helfrich functional and other building blocks that allow for a custom setup
or development of algorithms. This interface comes as a hierarchy of modules:
:mod:`trimem.core` is a compiled C++ module that encapsulates efficient
kernels for the evaluation of the Helfrich functional and its gradient. It
further provides data structures used for the consistent parametrization of the
Helfrich functional on the C++ as well as the python side. The subpackage
:mod:`trimem.mc` contains several building blocks (IO, algorithms, utilities)
based on the core module that can be used to compose versatile application
interfaces. Finally, :mod:`trimem.mc.util` provides high level routines to
run minimization and sampling algorithms directly.

A minimal showcase of a low-level usage of trimem is presented in the
following. For details on the available functionality, please refer to the
:ref:`API reference <modules>`.

Python example
--------------

As a preprocessing step, we start by generating a mesh using trimesh_:

.. _trimesh: https://trimsh.org/

>>> import trimesh
>>> mesh = trimesh.creation.icosphere(3)
>>> mesh.export('input.stl')

Such a mesh can then subsequently be used as an input to trimem:

>>> from trimem.core import read_mesh
>>> mesh = read_mesh('input.stl')

.. seealso::

   Prepocessiong routines are not part of trimem.
   :func:`trimem.core.read_mesh` therefore allows for the reading of
   meshes as offered by OpenMesh and returns a reference to
   :class:`TriMesh <trimem.core.TriMesh>` (the trimem-internal
   specialization of OpenMesh::TriMeshT). However, most high-level
   functionality in trimem is build around the python wrapper
   :class:`Mesh <trimem.mc.mesh.Mesh>`. This wrapper class can be directly
   generated from an input file via :func:`trimem.mc.mesh.read_trimesh`
   and contains a reference to ``TriMesh`` as the property ``trimesh``.

The Helfrich functional is represented in trimem by the
:class:`EnergyManager <trimem.core.EnergyManager>` that is parametrized
by :class:`EnergyParams <trimem.core.EnergyParams>`. ``EnergyParams`` is a
hierarchical data container that can be used, e.g., like this:

>>> from trimem.core import EnergyParams
>>> params = EnergyParams()
>>> params.kappa_b        = 30.0  # set bending modulus
>>> params.kappa_a        = 1.0e6 # set weight for surface area penalty
>>> params.kappa_v        = 1.0e6 # set weight for volume penalty
>>> params.kappa_c        = 1.0e6 # set weight for area difference penalty
>>> params.kappa_t        = 1.0e5 # set weight for tether regularization
>>> params.kappa_r        = 1.0e3 # set weight for surface repulsion
>>> params.area_frac      = 0.5   # target surface area as fraction of initial geometry
>>> params.volume_frac    = 1.0   # target volume as fraction of initial geometry
>>> params.curvature_frac = 1.0   # target curvature as fraction of initial geometry

Please refer to the :class:`documentation <trimem.core.EnergyParams>` for
a detailed list of parameters that can be set.

.. seealso::

   A convenient setup of the parametrization is also available from
   :func:`trimem.mc.util.setup_energy_manager` that constructs
   a ``Mesh`` and an ``EnergyManager`` from a :ref:`config-file`.

Subsequently, a reference to the ``EnergyManager`` can be constructed:

>>> estore = EnergyManager(mesh, params)

This class now allows for the evaluation of the Helfrich functional and its
gradient via

>>> estore.energy(mesh)
1000755.3602085959
>>> estore.gradient(mesh)
array([[-4.36273933e+03,  7.05905986e+03,  2.21934965e-12],
       [-5.84541963e+03,  7.64279670e+03,  6.93379067e+02],
       [-4.72350745e+03,  8.33617318e+03,  1.12191046e+03],
       ...,
       [ 9.73550747e+03,  7.90269602e+03,  9.67246248e+02],
       [ 9.73550747e+03,  7.90269602e+03, -9.67246248e+02],
       [ 7.62241050e+03,  7.90445464e+03, -1.11092383e-12]])




