.. _config-file:

Configuration file
==================

The trimem configuration file follows an INI-style layout and is internally
parsed by python's `configparser`_ module. It allows comments (also inline
comments) to be specified by both ``#`` and ``;``. The latter is used in the
following to indicate parameters that have meaningful default values if
omitted from the configuration file.

.. _configparser: https://docs.python.org/3/library/configparser.html

A verbosely commented default configuration file can be generated with the
help of the :ref:`cli` by running ``mc_app config``:

.. program-output:: mc_app config
