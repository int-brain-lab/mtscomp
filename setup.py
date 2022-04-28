# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import re

from setuptools import setup, find_packages
from typing import List


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

curdir = op.dirname(op.realpath(__file__))
with open(op.join(curdir, 'README.md')) as f:
    readme = f.read()


# Find version number from `__init__.py` without executing it.
filename = op.join(curdir, 'mtscomp.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


def get_requirements(extra: str = "") -> List[str]:
    requirements_file = f"requirements-{extra}.txt" if extra else "requirements.txt"
    requires = []
    with open(requirements_file) as f:
        for dep in f.readlines():
            if dep.startswith("git+"):
                print(f"Note! Dependency {dep} from file {requirements_file} was omitted.")
                continue
            requires.append(dep.strip())
    return requires


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
    py_modules=['mtscomp'],
    packages=find_packages(),
    package_dir={'mtscomp': 'mtscomp'},
    package_data={
        'mtscomp': [],
    },
    entry_points={
        'console_scripts': [
            'mtscomp=mtscomp:mtscomp',
            'mtsdecomp=mtscomp:mtsdecomp',
            'mtsdesc=mtscomp:mtsdesc',
            'mtschop=mtscomp:mtschop',
        ],
    },
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements("dev"),
    },
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
