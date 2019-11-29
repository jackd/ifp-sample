from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from Cython.Build import cythonize
import numpy

with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')

setup(
    name='ifp-sample',
    version='0.0.1',
    description='Cython iterative farthest point sampling implementation',
    url='https://github.com/jackd/ifp-sample.git',
    author='Dominic Jack',
    author_email='thedomjack@gmail.com',
    license='MIT',
    packages=['ifp'],
    requirements=install_requires,
    include_dirs=[numpy.get_include()],
    zip_safe=True,
    ext_modules=cythonize(["ifp/_ifp.pyx", "ifp/_rej.pyx", "ifp/_hybrid.pyx"]),
)
