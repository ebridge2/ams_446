from distutils.core import setup
from setuptools import setup
import clustering

setup(
    name='clustering',
    packages=[
        'clustering'
    ],
    description='A package for Spectral Clustering.',
    author='Eric Bridgeford and Theodor Marinov',
    author_email='ebridge2@jhu.edu, tmarino1@jhu.edu',
    keywords=[
        'clustering',
        'spectral',
        'kmeans'
    ],
    install_requires=[  # We didnt put versions for numpy, scipy, b/c travis-ci
        'numpy',  # We use nump v1.10.4
        'scipy',  # We use 0.17.0
        'matplotlib>=1.12'
    ]
)