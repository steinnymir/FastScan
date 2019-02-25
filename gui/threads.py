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

import os, sys
import time
from multiprocessing import Pool

import nidaqmx
import numpy as np
from PyQt5 import QtCore
from nidaqmx import stream_readers
from nidaqmx.constants import Edge, AcquisitionType

from utilities.data import bin_dc_multi, bin_dc, project_to_time_axis
from utilities.math import gaussian


class Thread(QtCore.QThread):
    stopped = QtCore.pyqtSignal()

    def stop(self):
        self.threadactive = False
        self.wait()
        self.stopped.emit()


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(OSError)

    def on_error(self, e):
        print('error in {} class'.format(type(self)))
        self.error.emit(e)


class Streamer(Worker):

    def __init__(self, n_samples, iterations=None, simulate=False, dark_control=True):
        super().__init__()

        self.n_samples = n_samples
        self.iterations = iterations
        self.data = np.zeros((3, n_samples))
        self.simulate = simulate
        self.dark_control = dark_control

        self.should_stop = True

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        if self.simulate:
            self.start_simulated_acquisition()
        else:
            try:
                with nidaqmx.Task() as task:

                    self.reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
                    print('reader initialized')
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  # shaker position chanel
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  # signal chanel
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai2")  # dark control chanel
                    print('added tasks')
                    task.timing.cfg_samp_clk_timing(1000000, source="/Dev1/PFI0",
                                                    active_edge=Edge.RISING,
                                                    sample_mode=AcquisitionType.CONTINUOUS)
                    task.start()
                    print('started tasks')

                    self.should_stop = False
                    i = 0
                    while True:
                        i += 1
                        print('measuring cycle {}'.format(i))
                        self.measure()
                        if self.iterations is not None and i >= self.iterations:
                            self.should_stop = True
                        if self.should_stop:
                            print('loop broken, acquisition stopped.')
                            self.finished.emit()
                            break

            except Exception as e:
                self.on_error(e)

    def start_simulated_acquisition(self):
        self.should_stop = False
        if self.iterations is None:
            i = 0
            while not self.should_stop:
                i += 1
                print('measuring cycle {}'.format(i))
                self.simulate_measure()
        else:
            for i in range(self.iterations):
                self.simulate_measure()

    @QtCore.pyqtSlot()
    def stop_acquisition(self):
        print('thread got message to stop')
        self.should_stop = True
        print('scheduled to stop')

    def measure(self):

        self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.n_samples)
        print('reader wrote data: {}'.format(self.data.mean()))
        self.newData.emit(self.data)

    def simulate_measure(self):
        t0 = time.time()
        n = np.arange(len(self.data[0]))
        phase = np.random.rand(1) * 2 * np.pi
        self.data[0, :] = np.cos(2*np.pi*n / 30000 + phase)
        self.data[2, :] = np.array([i % 2 for i in range(len(self.data[0]))])
        for i in range(len(n)):
            self.data[1, i] = self.data[0,i]/3 + 1*np.sin(np.random.rand(1))
            if self.data[2,i] ==1:
                self.data[1, i] += gaussian(self.data[0, i], 0, .1) + 1*np.random.rand(1)
        dt = time.time()-t0
        time.sleep(max(self.n_samples/273000 - dt,0))
        self.newData.emit(self.data)


class Binner(Worker):

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
        print('data binned: emitting results: {}\nprocessing time: {}'.format(type(self.data_output), time.time() - t0))
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
        print('data prepared, starting multiprocess')

        pool = Pool(chunks)
        results = pool.map(bin_dc_multi, args)
        results = np.array(results)

        binned_signal = []
        normarray = []

        for i in range(chunks):
            binned_signal.append(results[i, 0, :])
            normarray.append(results[i, 1, :])
        binned_signal = np.nansum(np.array(binned_signal), 0)
        normarray = np.nansum(np.array(normarray), 0)
        print('binned data : shape {}'.format(self.data_output.shape))
        self.data_output = binned_signal / normarray

    def bin_data(self):
        self.data_output = bin_dc(self.data_input, self.bins)


class Projector(Worker):

    def __init__(self, data, n_points,dark_control=True):
        super().__init__()
        self.data_input = data
        self.n_points = n_points
        self.data_output = np.zeros(n_points)
        self.dark_control = dark_control

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()
        try:
            self.project_data()
        except Exception as e:
            self.error.emit(e)
        print('data projected: emitting results: {}\nprocessing time: {}'.format(type(self.data_output),
                                                                                 time.time() - t0))
        self.newData.emit(self.data_output)
        self.finished.emit()

    def project_data(self):
        x, self.data_output = project_to_time_axis(self.data_input, self.n_points,dark_control=self.dark_control)


def main():
    pass


if __name__ == '__main__':
    pass
