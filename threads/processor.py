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
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import xarray as xr
from PyQt5 import QtCore

from threads.core import Worker
from utilities.data import bin_dc_multi, bin_dc


def main():
    pass


if __name__ == '__main__':
    main()


class Processor(Worker):
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, data, use_dark_control=True):

        super().__init__()
        self.logger = logging.getLogger('{}.Processor'.format(__name__))
        self.logger.debug('Created Processor')

        self.shaker_positions = data[0]
        self.signal = data[1]
        self.dark_control = data[2]

        self.use_dark_control = use_dark_control

        step = 0.000152587890625
        minpos = self.shaker_positions.min()
        min_t = (minpos / step) * .05  # consider 0.05 ps step size from shaker digitalized signal
        maxpos = self.shaker_positions.max()
        max_t = (maxpos / step) * .05

        n_points = int((maxpos - minpos) / step)

        self.position_bins = np.array((self.shaker_positions - minpos) / step, dtype=int)
        self.result = np.zeros(n_points + 1, dtype=np.float64)
        self.normamlization_array = np.zeros(n_points + 1, dtype=np.float64)

        self.output_array = np.zeros((2, n_points + 1))
        self.output_array[0] = np.linspace(min_t, max_t, n_points + 1)

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()

        try:
            if self.use_dark_control:
                self.project_dc()
            else:
                self.project()
            self.output_array[1] = self.result / self.normamlization_array

            self.newData.emit(self.output_array)
            self.logger.debug('Projected {} points to a {} pts array, with {} nans in : {:.2f} ms'.format(
                len(self.signal), len(self.result),
                len(self.result) - len(self.output_array[1][np.isfinite(self.output_array[1])]),
                1000 * (time.time() - t0)))
        except Exception as e:
            self.logger.warning(
                'failed to project data with shape {} to shape {}.\nERROR: {}'.format(self.shaker_positions.shape,
                                                                                      self.output_array.shape, e))
            self.error.emit(e)
        self.finished.emit()

    def project_dc(self):

        for i, pos in enumerate(self.position_bins):
            if self.dark_control[i]:
                self.result[pos] += self.signal[i]
                self.normamlization_array[pos] += 1.
            else:
                self.result[pos] -= self.signal[i]

    def project(self):
        for val, pos in zip(self.signal, self.position_bins):
            self.result[pos] += val
            self.normamlization_array[pos] += 1.


class Processor_multi(Worker):
    isReady = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(xr.DataArray)

    def __init__(self, id):
        super().__init__()
        self.logger = logging.getLogger('{}.Processor_multi'.format(__name__))
        self.logger.debug('Created Processor_multi: id={}'.format(id))

        self.id = id

    def initialize(self):
        self.logger.debug('created processor ID:{}'.format(self.id))
        self.isReady.emit(self.id)

    def project(self, n_points, signal, position_bins):
        result = np.zeros(n_points, dtype=np.float64)
        norm_array = np.zeros(n_points, dtype=np.float64)
        for val, pos in zip(signal, position_bins):
            result[pos] += val
            norm_array[pos] += 1.
        return result / norm_array

    def project_dc(self, n_points, signal, position_bins, dark_control):
        result = np.zeros(n_points, dtype=np.float64)
        norm_array = np.zeros(n_points, dtype=np.float64)
        for val, pos, dc in zip(signal, position_bins, dark_control):
            if dc:
                result[pos] += val
                norm_array[pos] += 1.
            else:
                result[pos] -= val

        return result / norm_array

    @QtCore.pyqtSlot()
    def work(self, stream_data, use_dark_control=True):
        """ project data from streamer format to 1d time trace

        creates bins from digitizing the stage positions measured channel of the
        stream data. Values from the signal channel are assigned to the corresponding
        bin from the stage positions. if Dark Control is true, values where dc
        is true are added, while where dark control is false, it is substracted.



        :param stream_data:
        :param use_dark_control:
        :return:
            xarray containing projected data and relative time scale.

        """
        t0 = time.time()
        shaker_positions = stream_data[0]
        signal = stream_data[1]
        dark_control = stream_data[2]

        step = 0.000152587890625  # ADC step size - corresponds to .25fs
        # consider 0.1 ps step size from shaker digitalized signal,
        # should be considering the 2 passes through the shaker
        step_to_time_factor = .1
        minpos = shaker_positions.min()
        min_t = (minpos / step) * step_to_time_factor
        maxpos = shaker_positions.max()
        max_t = (maxpos / step) * step_to_time_factor

        n_points = int((maxpos - minpos) / step) + 1
        time_axis = np.linspace(min_t, max_t, n_points)

        position_bins = np.array((shaker_positions - minpos) / step, dtype=int)

        try:
            if use_dark_control:
                result = self.project_dc(n_points, signal, position_bins, dark_control)
            else:
                result = self.project(n_points, signal, position_bins)

            output = xr.DataArray(result, coords={'time': time_axis}, dims='time')

            self.newData.emit(output)

            self.logger.debug('Projected {} points to a {} pts array, with {} nans in : {:.2f} ms'.format(
                len(signal), len(result),
                len(result) - len(result[np.isfinite(result)]),
                1000 * (time.time() - t0)))

        except Exception as e:
            self.logger.warning(
                'failed to project stream_data with shape {} to shape {}.\nERROR: {}'.format(shaker_positions.shape,
                                                                                             output.shape, e))
            self.error.emit(e)

        time.sleep(0.002)
        self.isReady.emit(self.id)
        self.logger.debug('Processor ID:{} is ready for new stream_data'.format(self.id))


class Binner(Worker):
    """DEPRECATED"""

    def __init__(self, data, n_points, multithreading=True):
        super().__init__()
        self.data_input = data
        self.bins = np.linspace(data[0].min(), data[0].max(), n_points)
        self.data_output = np.zeros_like(self.bins)
        self.multithreading = multithreading

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()
        try:
            if self.multithreading:
                self.bin_data_multi()
            else:
                self.bin_data()
        except Exception as e:
            self.error.emit(e)
        self.logger.debug('data binned: \n\tresults: {}\tprocessing time: {}'.format(self.data_output.shape,
                                                                                     time.time() - t0))
        self.newData.emit(self.data_output)
        self.finished.emit()

    def bin_data_multi(self):
        chunks = os.cpu_count()
        for i in range(os.cpu_count()):
            chunks = os.cpu_count() - 1 - i
            try:
                data_split = np.split(self.data_input, chunks, axis=1)
                break
            except ValueError:
                pass
        args = []
        for data in data_split:
            args.append((data, self.bins))
        self.logger.debug('data prepared for binning, starting multiprocess')

        pool = mp.Pool(chunks)
        results = pool.map(bin_dc_multi, args)
        results = np.array(results)

        binned_signal = []
        normarray = []

        for i in range(chunks):
            binned_signal.append(results[i, 0, :])
            normarray.append(results[i, 1, :])
        binned_signal = np.nansum(np.array(binned_signal), 0)
        normarray = np.nansum(np.array(normarray), 0)
        self.logger.debug('binned data : shape {}'.format(self.data_output.shape))
        self.data_output = binned_signal / normarray

    def bin_data(self):
        self.data_output = bin_dc(self.data_input, self.bins)
