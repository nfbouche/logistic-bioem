#!/usr/bin/env python
# coding=utf-8

# DISCLAIMER: I am not sure this will work properly for everyone.
#             Please report bugs.

# Run this script to leverage the power of `distutils` to install or build.

# Doc :
# - https://docs.python.org/2/install/index.html
# - https://docs.python.org/2/distutils/setupscript.html

import io
from setuptools import setup, find_packages

# Read version.py
__version__ = None
with io.open('logistic/__version__.py') as f:
    exec(f.read())

setup(
    name='logistic',
    version=__version__,
    author='Nicolas Bouché',
    author_email='nicolas.bouche@univ-lyon1.fr',
    description='A tool to reproduce the results from Bouché & McConway ',
    packages=find_packages(),
    package_data = { 'logistic': ['data/*tex']},
    install_requires=['astropy>=2.0', 'numpy>=1.14', 'scipy>=1.0.1,<1.2', 'matplotlib>=2.0', 'pandas>=0.18', 'pymc3==3.3', 'theano==1.0.4'],
)
