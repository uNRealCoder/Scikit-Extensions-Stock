from setuptools import find_namespace_packages, setup, find_packages
from codecs import open
from os import path

VERSION = '0.0.2'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

pkgs = []
pkgDict = {}
def addPackage():
    global pkgs
    global pkgDict
    pkgDict['scikit_extensions_Stock'] = path.join('scikit_extensions_Stock')
    pkgs.append('scikit_extensions_Stock.*')

addPackage()

setup(
    name='sklearn-extensions-stock',
    version=VERSION,
    description='A bundle of 3rd party extensions to scikit-learn',
    long_description=long_description,
    url='',
    license='MIT',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='scikit-learn sklearn extensions machine learning Stock',
    packages= find_packages(),
    include_package_data=False,
    author='Nikhil Ranjan',
    install_requires=install_requires,
    author_email='Nikhil.Ranjan@uga.edu'
)
#---------------------------------------------------------------------------

# import sys
# import os

# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration

#     config = Configuration('sklearn', parent_package, top_path)
#     config.name = 'Sklearn-Extenstions-Stock'
#     config.version = VERSION
#     # submodules with build utilities
#     config.add_subpackage('lstm',r'D:\UGA\Research\Stock\Scikit-Extensions-Stock\Scikit_Extensions_For_Stocks\lstm')
#     config.add_subpackage('sliding_window_transformer',r'D:\UGA\Research\Stock\Scikit-Extensions-Stock\Scikit_Extensions_For_Stocks\sliding_window_tranformer')
#     config.add_subpackage('time_series_scaler',r'D:\UGA\Research\Stock\Scikit-Extensions-Stock\Scikit_Extensions_For_Stocks\time_series_scaler')
#     # add the test directory
#     config.add_subpackage('tests')
#     config.
#     return config


# if __name__ == '__main__':
#     from numpy.distutils.core import setup
#     setup(**configuration(top_path='').todict(),
#     description='A bundle of 3rd party extensions to scikit-learn',
#     long_description='',
#     url='',
#     license='MIT')
# Test
