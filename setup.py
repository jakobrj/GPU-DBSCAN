from distutils.core import setup as CySetup
from distutils.core import Extension
# from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

DBSCAN_GPU = Extension(
    'dbscan_gpu',
    [
        'src/cython/DBSCAN_GPU.pyx',
    ],
    libraries=['DBSCAN_GPU'],
    library_dirs=[
        'cython',
        '.',
        '..',
    ],
    language='c++',
    include_dirs=[
        numpy.get_include(),
    ],
    runtime_library_dirs=['.']
)

CySetup(
# setup(
    name='dbscan_gpu',
    ext_modules=cythonize([DBSCAN_GPU])
)