import sys

from setuptools import find_packages
from skbuild import setup


setup(
    name="helfrich",
    version="0.0.1",
    description='Evaluation of the helfrich bending energy using OpenMesh',
    author='Sebastian Kehl',
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points="""
        [console_scripts]
        mc_app=helfrich.mc.app:cli
    """,
    cmake_install_dir="src/helfrich",
    include_package_data=True,
)
