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
import time

import numpy as np
from PyQt5 import QtCore
from scipy.optimize import curve_fit

from threads.core import Worker
from utilities.math import sech2_fwhm, gaussian_fwhm


def main():
    pass


if __name__ == '__main__':
    main()


class Fitter(Worker):

    def __init__(self, x, y, model='sech2', guess=None):
        super().__init__()
        self.x = x
        self.y = y
        if model == 'sech2':
            self.model = sech2_fwhm
        elif model in 'gaussian':
            self.model = gaussian_fwhm
        if guess is None:
            A = max(self.y)
            x0 = x[y.argmax()]
            fwhm = 1e-13
            offset = y[~np.isnan(y)].mean()
            self.guess = (A, x0, fwhm, offset)
        else:
            self.guess = guess

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()
        try:
            self.popt, pcov = curve_fit(self.model, self.x[np.isfinite(self.y)], self.y[np.isfinite(self.y)],
                                        p0=self.guess)
            self.newData.emit(self.popt)

        except Exception as e:
            self.error.emit(e)
            self.newData.emit(self.popt)

        self.finished.emit()