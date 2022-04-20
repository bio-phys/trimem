.. trimem documentation master file, created by
   sphinx-quickstart on Fri Apr  8 09:18:08 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to trimem's documentation
=================================

**Trimem** is a python package for the Monte Carlo simulation of lipid
membranes according to the Helfrich theory [Helfrich1973]_.

.. _installation:

Installation
------------

Trimem can be installed via

.. code-block:: Bash

   git clone --recurse-submodules https://gitlab.mpcdf.mpg.de/MPIBP-Hummer/trimem.git
   pip install trimem/

We suggest the installation using the ``--user`` flag to ``pip``. Alternatively,
we recommend to consider the usage of virtual environments to isolate the
installation of trimem, see, e.g., `here`_.

.. _here: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment

Dependencies
^^^^^^^^^^^^

Trimem builds upon the generic mesh data structure
`OpenMesh <https://www.graphics.rwth-aachen.de/software/openmesh/>`_, which
is included as a submodule that is pulled in upon ``git clone`` via the
``--recurse-submodules`` flag, see :ref:`installation`.

For the efficient exploitation of shared-memory parallelism, trimem makes
use of the `OpenMP <https://www.openmp.org/>`_ application programming model
(``>= v4.5``) and modern ``C++``. It thus requires relatively up-to-date
compilers (supporting at least ``C++17``).

If not already available, the following python dependencies will be
automatically installed:

* numpy
* scipy
* h5py
* meshio

Documentation and tests further require:

* autograd
* meshzoo
* sphinx
* sphinx-copybutton
* sphinxcontrib-programoutput

Usage
-----

**Trimem** provides a convenient user interface via a :ref:`cli` that is
available after :ref:`installation` under the respective ``bin`` directory
available to ``pip``. For a quick-start and more detailed documentation,
please have a look at the :ref:`usage`.

Alternatively, trimem can also be used as a
:ref:`python library <python-module>`. High level functionality is exposed in
the module :mod:`trimem.mc.util`. For details on low level functionality,
please refer to the :ref:`API reference <modules>`.

Citation
--------

If you use trimem for your scientific work, please consider the citation of
[Siggel2022]_.

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   modules

* :ref:`bibliography`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
