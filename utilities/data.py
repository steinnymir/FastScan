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
from multiprocessing import Pool

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

def project_to_time_axis(data,n_points,dark_control=True):
    minpos = data[0].min()
    maxpos = data[0].max()
    pos_bin = np.array((n_points-1)*(data[0] - minpos)/(maxpos-minpos),dtype=int)
    result = np.zeros(n_points)
    normarray = np.zeros(n_points,dtype=int)
    if dark_control:
        for i in range(len(pos_bin)):
            if data[2,i] == 1:
                result[pos_bin[i]] += data[1,i]
                normarray[pos_bin[i]] +=1
            if data[2,i] == 0:
                result[pos_bin[i]] -= data[1,i]
    else:
        for val,p,dc in zip(data[1],pos_bin,data[2]):
            if dc==1:
                result[p] += val
                normarray[p]+=1
    norm_res = result/normarray
    x_axis = np.linspace(minpos,maxpos,n_points)
    return x_axis, norm_res

def main():
    pass


if __name__ == '__main__':
    main()