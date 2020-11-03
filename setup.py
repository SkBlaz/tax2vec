## install the lib

## Py3plex installation file. Cython code for fa2 is the courtesy of Bhargav Chippada.
## https://github.com/bhargavchippada/forceatlas2

from os import path
from setuptools import setup, find_packages


def parse_requirements(file):
    required_packages = []
    with open(path.join(path.dirname(__file__), file)) as req_file:
        for line in req_file:
            required_packages.append(line.strip())
    return required_packages


setup(name='tax2vec',
      version='0.25',
      description="Semantic space vectorization algorithm",
      url='http://github.com/skblaz/tax2vec',
      author='Blaž Škrlj',
      author_email='blaz.skrlj@ijs.si',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=parse_requirements("requirements.txt"),
      include_package_data=True)
