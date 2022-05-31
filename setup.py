from distutils.core import setup as CySetup
from distutils.core import Extension
from Cython.Build import cythonize
import numpy

DBSCAN_GPU = Extension(
    'DBSCAN_GPU',
    [
        'src/cython/DBSCAN_GPU.pyx',
    ],
    libraries=['DBSCAN_GPU'],
    library_dirs=[
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
    name='DBSCAN_GPU',
    ext_modules=cythonize([DBSCAN_GPU])
)