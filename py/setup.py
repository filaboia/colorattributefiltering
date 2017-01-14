#!/usr/bin/env python

from distutils.core import setup

long_description = '''colorsegmentation is a framework for color attribute filtering oriented morphologic segmentation.'''

setup(name='colorsegmentation',
    version='1.0',
    description='colorsegmentation framework',
    long_description=long_description,
    author='Sergio Grijo de Sousa Filho',
    author_email='filaboia@gmail.com',
    url='http://www.github.com/filaboia/colorsegmentation',
    packages=['colorsegmentation', 'colorsegmentation.image', 'colorsegmentation.maxtree'],
)
