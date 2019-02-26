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

def gaussian(x, x0, sig):
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.)))

def sech2_fwhm(x, A, x0, fwhm,c):
    tau = fwhm*2/1.76
    return A / (np.cosh((x-x0)/tau))**2+c

def gaussian_fwhm(x, A,x0, fwhm,c):
    sig = fwhm*2/2.355
    return A*np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.)))+c

def sin(x,A,f,p):
    return A* np.sin(x/f + p)




def main():
    pass


if __name__ == '__main__':
    main()