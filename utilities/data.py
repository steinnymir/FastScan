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
import os
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool
from utilities.math import gaussian_fwhm, sech2_fwhm
def bin_dc(data,bins):
    binned_signal = np.zeros_like(bins)
    normarray = np.zeros_like(bins)
    for i in range(data.shape[1]//2):
        ii = 2*i
        j = (np.abs(bins - data[0][ii])).argmin()
        binned_signal[j] += (data[1][ii]-data[1][ii+1])
        normarray[j]+=1
    return binned_signal/normarray

def bin(data,bins):
    binned_signal = np.zeros_like(bins)
    normarray = np.zeros_like(bins)
    for i in range(data.shape[1]):
        j = (np.abs(bins - data[0][i])).argmin()
        binned_signal[j] += data[1][i]
        normarray[j]+=1
    return binned_signal/normarray


def bin_dc_multi(arguments):
    data, bins = arguments
    binned_signal = np.zeros_like(bins)
    normarray = np.zeros_like(bins)
    for i in range(data.shape[1]//2):
        ii = 2*i
        j = (np.abs(bins - data[0][ii])).argmin()
        binned_signal[j] += (data[1][ii]-data[1][ii+1])
        normarray[j]+=1
    return binned_signal, normarray

def bin_multi(arguments):
    data, bins = arguments
    binned_signal = np.zeros_like(bins)
    normarray = np.zeros_like(bins)
    for i in range(data.shape[1]):
        j = (np.abs(bins - data[0][i])).argmin()
        binned_signal[j] += data[1][i]
        normarray[j]+=1
    return binned_signal, normarray


def fit_peak(x,y,model='sech2',guess=(1,0,1e-13,.1)):
    if model == 'sech2':
        f = sech2_fwhm
    elif model == 'gauss':
        f = gaussian_fwhm
    print('guess: {}'.format(guess))
    popt,pcov = curve_fit(f, x,y, p0=guess)
    yf = f(x,*popt)
    return yf, popt

def main():
    pass


if __name__ == '__main__':
    main()