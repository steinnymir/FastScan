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
from multiprocessing import Pool
import time

import nidaqmx
import numpy as np
from nidaqmx import stream_readers
from nidaqmx.constants import Edge, AcquisitionType
from PyQt5 import QtCore

from utilities.data import bin_dc_multi, bin_dc

class Thread(QtCore.QThread):
    stopped = QtCore.pyqtSignal()
    def stop(self):
        self.threadactive = False
        self.wait()
        self.stopped.emit()

class Streamer(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(OSError)

    def __init__(self, n_samples, iterations=None, simulate=False):
        super().__init__()

        self.n_samples = n_samples
        self.iterations = iterations
        self.data = np.zeros((3, n_samples))
        self.simulate = simulate

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
                        if self.iterations is not None and i>= self.iterations:
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
        time.sleep(2 * self.n_samples / 300000)
        n = np.arange(len(self.data[0]))
        phase = np.random.rand(1) * 2 * np.pi
        self.data[0, :] = np.cos(n / 10000 + phase)
        self.data[1, :] = self.data[0, :] / 3
        self.data[2, :] = np.array([i % 2 for i in range(len(self.data[0]))])
        self.newData.emit(self.data)

    def on_error(self, e):
        self.error.emit(e)


class Binner(QtCore.QObject):
    newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(OSError)

    def __init__(self, data, bins, multithreading=True):
        super().__init__()
        self.data_input = data
        self.bins = bins
        self.data_output = np.zeros_like(bins)
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

    def bin_data_multi(self):


        chunks = os.cpu_count()
        for i in range(os.cpu_count()):
            chunks = os.cpu_count()-1-i
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

        self.data_output = bin_dc(self.data_input,self.bins)





def main():
    pass


if __name__ == '__main__':
    main()
