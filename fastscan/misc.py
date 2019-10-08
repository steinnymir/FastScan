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
from configparser import ConfigParser

import h5py
import numpy as np
import xarray as xr
from PyQt5 import QtWidgets


# -------------------------
# Qt stuff
# -------------------------
def my_exception_hook(exctype, value, traceback):
    """error catching for qt in pycharm"""
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


def labeledQwidget(label, widget, align='h'):
    """Create a horizontally aligned label and widget."""
    w = QtWidgets.QWidget()

    if align in ['h', 'horizontal']:
        l = QtWidgets.QHBoxLayout()
    elif align in ['vertical', 'v']:
        l = QtWidgets.QVBoxLayout()
    else:
        raise Exception('{} not a valid alignment, please use "h" or "v"'.format(align))
    l.addWidget(QtWidgets.QLabel(label))
    l.addWidget(widget)
    l.addStretch()
    w.setLayout(l)
    return w


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


def sech2_fwhm_wings(x, a, xc, fwhm, off, wing_sep, wing_ratio, wings_n):
    """ sech squared with n wings."""
    res = sech2_fwhm(x, a, xc, fwhm, off)
    for n in range(1, wings_n):
        res += sech2_fwhm(x, a * (wing_ratio ** n), xc - n * wing_sep, fwhm, 0)
        res += sech2_fwhm(x, a * (wing_ratio ** n), xc + n * wing_sep, fwhm, 0)

    return res


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_fwhm(x, A, x0, fwhm, c):
    sig = fwhm * 2 / 2.355
    return A * np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.))) + c


def transient_1expdec(t, A1, tau1, sigma, y0, off, t0):
    """ Fitting function for transients, 1 exponent decay.
    A: Amplitude
    Tau: exp decay
    sigma: pump pulse duration
    y0: whole curve offset
    off: slow dynamics offset"""
    from scipy.special import erf
    t = t - t0
    tmp = erf((sigma ** 2. - 5.545 * tau1 * t) / (2.7726 * sigma * tau1))
    tmp = .5 * (1 - tmp) * np.exp(sigma ** 2. / (11.09 * tau1 ** 2.))
    return y0 + tmp * (A1 * (np.exp(-t / tau1)) + off)


def read_h5(file):
    dd = {}
    with h5py.File(file, 'r') as f:
        if 'avg' in f:
            data = f['avg']['data']
            time = f['avg']['time_axis']
            dd['avg'] = xr.DataArray(data, coords={'time': time}, dims=('time'))
        if 'raw' in f:
            dd['raw'] = f['raw']['avg'][:]
        if 'all_data' in f:
            data = f['all_data']['data']
            time = f['all_data']['time_axis']
            dd['all_data'] = xr.DataArray(data, coords={'time': time}, dims=('avg', 'time'))
    return dd


# -------------------------
# exceptions
# -------------------------
class NoDataException(Exception):
    pass
