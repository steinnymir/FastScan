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
import numpy as np

def project_r0(spos,signal,dark_control,reference,use_dark_control):
    # cdef int n_pts, pos_min, res_size, i, pos
    # cdef float r0, val,ref

    n_pts = len(spos)
    pos_min = spos.min()
    res_size = spos.max() - pos_min + 1
    result_val = np.zeros(res_size)
    result_ref = np.zeros(res_size)
    norm_array = np.zeros(res_size)

    if use_dark_control:
        for i in range(n_pts//2):
            pos = int((spos[2 * i] + spos[2 * i + 1]) // 2) - pos_min
            if dark_control[2 * i] > dark_control[2 * i + 1]:
                val = signal[2 * i] - signal[2 * i + 1]
                ref = reference[2 * i]
            else:
                val = signal[2 * i + 1] - signal[2 * i]
                ref = reference[2 * i + 1]

            result_val[pos] += val
            result_ref[pos] += ref
            norm_array[pos] += 1

    else:

        for i in range(n_pts):
            result_val[spos[i]-pos_min] += signal[i]
            result_ref[spos[i]-pos_min] += reference[i]
            norm_array[spos[i]-pos_min] += 1.

    r0 = np.mean(result_ref/norm_array)
    return result_val/(norm_array*r0)


def project(spos,signal,dark_control,use_dark_control):
    n_pts = len(spos)
    pos_min = spos.min()
    res_size = spos.max() - pos_min + 1
    result_val = np.zeros(res_size)
    norm_array = np.zeros(res_size)


    if use_dark_control:
        for i in range(n_pts//2):
            pos = int((spos[2 * i] + spos[2 * i + 1]) // 2) - pos_min
            if dark_control[2 * i] > dark_control[2 * i + 1]:
                val = signal[2 * i] - signal[2 * i + 1]
            else:
                val = signal[2 * i + 1] - signal[2 * i]

            result_val[pos] += val
            norm_array[pos] += 1

    else:

        for i in range(n_pts):
            result_val[spos[i]-pos_min] += signal[i]
            norm_array[spos[i]-pos_min] += 1.

    return result_val/(norm_array)

if __name__ == '__main__':
    pass