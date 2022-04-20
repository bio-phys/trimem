import sys

from setuptools import find_packages
from skbuild import setup


setup(
    name="trimem",
    version="0.0.1",
    description='Evaluation of the Helfrich bending energy using OpenMesh',
    author='Sebastian Kehl',
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points="""
        [console_scripts]
        mc_app=trimem.mc.app:cli
    """,
    cmake_install_dir="src/trimem",
    include_package_data=True,
)
