#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


def calculate_version():
    initpy = open('tpotclustering/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version


package_version = calculate_version()

setup(
    name='TPOTClustering',
    version=package_version,
    author='Matheus Camilo da Silva, Sylvio Barbon Junior',
    author_email='matheuscmilo@gmail.com',
    packages=find_packages(),
    url='https://github.com/Mcamilo/tpot-clustering',
    license='GNU/LGPLv3',
    entry_points={'console_scripts': ['tpot-clustering=tpot:main', ]},
    description=('Tree-based Pipeline Optimization Tool'),
    long_description='''
    TPOT-Clustering is a Python tool that automatically creates and optimizes unsupervised machine learning (clustering) pipelines using genetic programming.
    It is a fork of the original TPOT project, extended to support clustering tasks.

    Contact
    =============
    If you have any questions or comments about TPOT-Clustering, please feel free to contact:

    E-mail: matheus.camilo@phd.units.it or sylvio.barbon@units.it

    This project is hosted at https://github.com/Mcamilo/tpot-clustering  
    Original TPOT project: https://github.com/EpistasisLab/tpot
    ''',

    zip_safe=True,
    install_requires=['numpy>=1.16.3',
                    'scipy>=1.3.1',
                    'scikit-learn>=1.4.1',
                    'deap>=1.2',
                    'update_checker>=0.16',
                    'tqdm>=4.36.1',
                    'stopit>=1.1.1',
                    'pandas>=0.24.2',
                    'joblib>=0.13.2'],
    extras_require={
        'skrebate': ['skrebate>=0.3.4'],
        'mdr': ['scikit-mdr>=0.4.4'],
        'dask': ['dask>=0.18.2',
                 'distributed>=1.22.1',
                 'dask-ml>=1.0.0'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['clustering', 'pipeline optimization', 'hyperparameter optimization', 'data science', 'machine learning', 'genetic programming', 'evolutionary computation'],
)
