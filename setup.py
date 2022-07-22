from setuptools import find_packages
from skbuild import setup

# minimal version of setup() as still required by scikit-build for interception
setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        'console_scripts': [
            'mc_app=trimem.mc.app:cli',
        ]
    },
    cmake_install_dir="src/trimem",
    include_package_data=True,
    scripts=["src/debug/testd"],
)
