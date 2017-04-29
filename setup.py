from distutils.core import setup
from setuptools import setup
import clustering

VERSION = ndmg.version

setup(
    name='clustering',
    packages=[
        'clustering',
        'clustering.spectral',
        'clustering.kmeans'
    ],
    version=VERSION,
    description='A package for Spectral Clustering.',
    author='Eric Bridgeford and Theodor Marinov',
    author_email='ebridge2@jhu.edu, tmarino1@jhu.edu',
    keywords=[
        'clustering',
        'spectral',
        'kmeans'
    ],
    classifiers=[],
    install_requires=[  # We didnt put versions for numpy, scipy, b/c travis-ci
        'numpy',  # We use nump v1.10.4
        'scipy',  # We use 0.17.0
        'matplotlib>=1.12'
    ]
)