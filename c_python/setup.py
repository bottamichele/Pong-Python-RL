import numpy as np

from distutils.core import setup, Extension

#Modules
sumtree_module = Extension("sum_tree", ["./c_python/src/sum_tree.c"], [np.get_include()])

#Setup
setup(ext_modules=[sumtree_module])