# setup.py

# Imports
import os
import sys

from os.path import join
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

# Numpy from http://www.scipy.org/Cookbook/SWIG_NumPy_examples
import numpy as np

try:
    NUMPY_INCLUDE = np.get_include()
except AttributeError:
    NUMPY_INCLUDE = np.get_numpy_include()

# `EIGEN_INCLUDE` and `UTIL_CPP_INCLUDE` from site.cfg.
import ConfigParser
c = ConfigParser.ConfigParser()
# Preserve case. See:
# http://stackoverflow.com/questions/1611799/preserve-case-in-configparser
c.optionxform = str
c.read('site.cfg')
EIGEN_INCLUDE = c.get('Include', 'EIGEN_INCLUDE')
UTIL_CPP_INCLUDE = c.get('Include', 'UTIL_CPP_INCLUDE')

# Setup.
qpbo_dir = 'external/QPBO-v1.32.src/' 
include_dirs = [NUMPY_INCLUDE, EIGEN_INCLUDE, UTIL_CPP_INCLUDE,
                'include/', qpbo_dir,
                '.']

qpbo_srcs = [join(qpbo_dir, f) for f in os.listdir(qpbo_dir)
             if os.path.splitext(f)[1] == '.cpp']

setup(ext_modules=[
        Extension('qpbo_alpha_expand.qpbo_alpha_expand',
                  ['qpbo_alpha_expand/qpbo_alpha_expand.pyx'] + qpbo_srcs,
                  language='c++',
                  include_dirs=include_dirs),
      ],
      cmdclass = {'build_ext' : build_ext})