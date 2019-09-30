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
import ast
import os
import sys
import numpy as np
from configparser import ConfigParser


# -------------------------
# Qt error catching
# -------------------------
def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


# -------------------------
# Settings persing
# -------------------------
def parse_category(category, settings_file='default'):
    """ parse setting file and return desired value

    Args:
        category (str): title of the category
        setting_file (str): path to setting file. If set to 'default' it takes
            a file called SETTINGS.ini in the main folder of the repo.

    Returns:
        dictionary containing name and value of all entries present in this
        category.
    """
    settings = ConfigParser()
    if settings_file == 'default':
        current_path = os.path.dirname(__file__)
        while not os.path.isfile(os.path.join(current_path, 'SETTINGS.ini')):
            current_path = os.path.split(current_path)[0]

        settings_file = os.path.join(current_path, 'SETTINGS.ini')
    settings.read(settings_file)
    try:
        cat_dict = {}
        for k, v in settings[category].items():
            try:
                cat_dict[k] = ast.literal_eval(v)
            except ValueError:
                cat_dict[k] = v
        return cat_dict
    except KeyError:
        print('No category "{}" found in SETTINGS.ini'.format(category))


def parse_setting(category, name, settings_file='default'):
    """ parse setting file and return desired value

    Args:
        category (str): title of the category
        name (str): name of the parameter
        setting_file (str): path to setting file. If set to 'default' it takes
            a file called SETTINGS.ini in the main folder of the repo.

    Returns:
        value of the parameter, None if parameter cannot be found.
    """
    settings = ConfigParser()
    if settings_file == 'default':
        current_path = os.path.dirname(__file__)
        while not os.path.isfile(os.path.join(current_path, 'SETTINGS.ini')):
            current_path = os.path.split(current_path)[0]

        settings_file = os.path.join(current_path, 'SETTINGS.ini')
    settings.read(settings_file)

    try:
        value = settings[category][name]
        return ast.literal_eval(value)
    except KeyError:
        print('No entry "{}" in category "{}" found in SETTINGS.ini'.format(name, category))
        return None
    except ValueError:
        return settings[category][name]
    except SyntaxError:
        return settings[category][name]


def write_setting(value, category, name, settings_file='default'):
    """ Write enrty in the settings file

    Args:
        category (str): title of the category
        name (str): name of the parameter
        setting_file (str): path to setting file. If set to 'default' it takes
            a file called SETTINGS.ini in the main folder of the repo.

    Returns:
        value of the parameter, None if parameter cannot be found.
    """
    settings = ConfigParser()
    if settings_file == 'default':
        current_path = os.path.dirname(__file__)
        while not os.path.isfile(os.path.join(current_path, 'SETTINGS.ini')):
            current_path = os.path.split(current_path)[0]

        settings_file = os.path.join(current_path, 'SETTINGS.ini')
    settings.read(settings_file)

    settings[category][name] = str(value)

    with open(settings_file, 'w') as configfile:
        settings.write(configfile)


# -------------------------
# math
# -------------------------

def update_average(new, avg, n):
    'recalculate average with new dataset.'
    prev_n = (n - 1) / n
    # return (avg  * (n - 1) + new) / n
    return avg * prev_n + new / n


def sin(x, A, f, p, o):
    return A * np.sin(x / f + p) + o


def sech2_fwhm(x, A, x0, fwhm, c):
    tau = fwhm * 2 / 1.76
    return A / (np.cosh((x - x0) / tau)) ** 2 + c

# -------------------------
# exceptions
# -------------------------
class NoDataException(Exception):
    pass