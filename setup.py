"""
*****************************************************
*             Filter Pruning
*
*              Gabor Vecsei
* Website:     https://gaborvecsei.com
* Blog:        https://gaborvecsei.wordpress.com/
* LinkedIn:    https://www.linkedin.com/in/gaborvecsei
* Github:      https://github.com/gaborvecsei
*
*****************************************************
"""

from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ridurre',
    version='0.0.2',
    description='Keras model convoltuianl filter pruning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gaborvecsei.com',
    author='Gabor Vecsei',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'],
    keywords='pruning keras vecsei gaborvecsei filter-pruning convolution deep-learning machine-learning prune ridurre',
    packages=find_packages(),
)
