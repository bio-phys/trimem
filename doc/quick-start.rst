.. _quick-start:

Quick-Start
===========

This tutorial shows the Markov Chain Monte Carlo simulation of a unit-sphere
subject to a volume reduction of 55% under constant surface area. To this end,
we will run a precursor minimization of the energy to facilitate the subsequent
sampling of the equilibrium configuration.

Mesh generation
---------------

For the reading of mesh geometries, trimem uses the IO facilities provided by
OpenMesh_ and thus supports IO-formats as provided by OpenMesh_. For high
quality meshes of simple geometries we make use of the meshzoo_ python library
in this tutorial using the `stl`-format.

.. _OpenMesh: https://www.graphics.rwth-aachen.de/software/openmesh/
.. _meshzoo: https://pypi.org/project/meshzoo/

>>> import meshzoo
>>> import meshio
>>> points, cells = meshzoo.icosa_sphere(8);
>>> meshio.write_points_cells('input.stl', points, [('triangle', cells)])

Configuration
-------------

We start by defining a :ref:`config-file`. It defines the setup of the energy
in the section ``[ENERGY]``. The ``algorithm`` is set to ``minimize`` to
reflect that we are going to start by a precursor minimization that is defined
in the section ``[MINIMZATION]``. The additional tether regularization and the
repulsion constraint are configured in the sections ``[BONDS]`` and
``[SURFACEREPULSION]``, respectively.

.. code-block:: Bash

   cat << EOF > inp.conf
   [GENERAL]
   algorithm = minimize
   info = 100
   input = input.stl
   output_format = vtu
   [BONDS]
   bond_type = Edge
   r = 2
   [SURFACEREPULSION]
   n_search = cell-list
   rlist = 0.2
   exclusion_level = 2
   refresh = 10
   r = 2
   lc1 = 0.15
   [ENERGY]
   kappa_b = 300.0
   kappa_a = 1.0e6
   kappa_v = 1.0e6
   kappa_c = 0.0
   kappa_t = 1.0e5
   kappa_r = 1.0e3
   area_fraction = 1.0
   volume_fraction = 0.45
   curvature_fraction = 1.0
   continuation_delta = 0.0
   continuation_lambda = 1.0
   [HMC]
   num_steps = 10000
   traj_steps = 100
   step_size = 2.5e-5
   momentum_variance = 1.0
   thin = 100
   flip_ratio = 0.1
   flip_type = parallel
   initial_temperature = 1.0
   cooling_factor = 1.0e-3
   start_cooling = 10000
   [MINIMIZATION]
   maxiter = 2000
   out_every = 0
   EOF

.. seealso::

   A verbosely commented default configuration can be generated with
   ``mc_app config``, see :ref:`cli`.

Minimization
------------

We can now run the precursor minimization with

.. code-block:: Bash

   mc_app run --conf inp.conf

This will write the result of the minimization as `vtu`-file. Additionally, a
automatically numbered (starting from 0) checkpoint file is written that can
be used to restart the subsequent Monte Carlo sampling.

Sampling
--------

To restart a sampling simulation from a precursor minimization, the
``algorithm`` first has to be changed to ``hmc`` in the configuration file:

.. code-block:: Bash

   sed -i 's/= minimize/= hmc/g' inp.conf

The sampling can then be initiated from the checkpoint file written by the
minimization step with:

.. code-block:: Bash

   mc_app run --conf inp.conf --restart

This will write a series of `vtu`-files representing the trajectory of the
simulated Markov Chain (output frequency controlled by the ``thin`` parameter
of the ``[HMC]`` section).

.. seealso::

   Besides the `vtu` output, trimem support also `xyz` and `xdmf` formats.
   See :mod:`trimem.mc.output`.
