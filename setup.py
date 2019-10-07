# -*- coding: utf-8 -*-
"""

@author: Steinn Ymir Agustsson

    Copyright (C) 2018 Steinn Ymir Agustsson

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os
from configparser import ConfigParser

if not os.path.isfile('SETTINGS.ini'):
    print('creating settings file')
    cp = ConfigParser()
    cp.read('Default_SETTINGS.ini')
    print(cp.sections())
    with open('SETTINGS.ini','w+') as settingsFile:
        cp.write(settingsFile)
else:
    print('settings file found')

extensions = [
    Extension("fastscan.cscripts.project", [os.path.join("fastscan", "cscripts", "project.pyx")],
        include_dirs=[numpy.get_include()]),
]

setup(
    name="FastScan",
    version='0.1.0',
    description='Fast-scan Pump Probe software',
    author=['Steinn Ymir Agustsson'],
    url='https://github.com/steinnymir/FastScan',
    packages=['distutils', 'distutils.command'],
    ext_modules=cythonize(extensions)
)
