"""Trimem.

Utilities and algorithms for the Helfrich bending energy. It comprises
a C++ submodule for the heavy lifting as well as a python module for
convenient algorithm development.
"""

try:
    from ._version import version as __version__
except:
    __version__ = "unknown"
