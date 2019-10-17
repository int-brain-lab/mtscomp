# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


curdir = op.dirname(op.realpath(__file__))
with open(op.join(curdir, 'README.md')) as f:
    readme = f.read()


# Find version number from `__init__.py` without executing it.
filename = op.join(curdir, 'mtscomp.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='mtscomp',
    version=version,
    license="BSD",
    description='Lossless compression for electrophysiology time-series',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Cyrille Rossant (International Brain Laboratory)',
    author_email='cyrille.rossant@gmail.com',
    url='https://github.com/int-brain-lab/mtscomp',
    packages=_package_tree('mtscomp'),
    package_dir={'mtscomp': 'mtscomp'},
    package_data={
        'mtscomp': [],
    },
    entry_points={
        'console_scripts': [
            'mtscomp=mtscomp:mtscomp',
            'mtsdecomp=mtscomp:mtsdecomp',
            'mtsdesc=mtscomp:mtsdesc',
        ],
    },
    include_package_data=True,
    keywords='lossless,compresssion,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
