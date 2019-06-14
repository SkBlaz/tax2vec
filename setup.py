## install the lib


## Py3plex installation file. Cython code for fa2 is the courtesy of Bhargav Chippada.
## https://github.com/bhargavchippada/forceatlas2

from os import path
import sys
from setuptools import setup,find_packages
from setuptools.extension import Extension
    
setup(name='tax2vec',
      version='0.22',
      description="Semantic space vectorization algorithm",
      url='http://github.com/skblaz/tax2vec',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['rdflib','numpy','networkx','scipy','sklearn'],
      include_package_data=True)

