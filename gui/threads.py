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

import multiprocessing as mp
import queue
from scipy.optimize import curve_fit

try:
    import nidaqmx
    from nidaqmx import stream_readers
    from nidaqmx.constants import Edge, AcquisitionType
except:
    print('no nidaqmx package installed.')

import numpy as np
from PyQt5 import QtCore

from utilities.data import bin_dc_multi, bin_dc
from utilities.math import gaussian, gaussian_fwhm, sech2_fwhm


class Thread(QtCore.QThread):
    stopped = QtCore.pyqtSignal()

    def stop(self):
        self.threadactive = False
        self.wait()
        self.stopped.emit()


class Worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(Exception)

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
        noise = np.random.rand(len(n))
        phase = noise[0] * 2 * np.pi

        amplitude = .5*(1+.02*np.random.uniform(-1,1))
        self.data[0, :] =  np.cos(2*np.pi*n / 30000 + phase)*amplitude/2 #in volt

        # self.data[2, :] = np.array([i % 2 for i in range(len(n))])
        for i in range(len(n)):
            self.data[2, i] = i % 2 # dark control channel filled with 1010...
            self.data[1, i] = self.data[0,i]/3 + 1*np.sin(noise[2])
            if i % 2 == 1:
                self.data[1, i] += gaussian(self.data[0, i], 0, .01) + noise[i]

        dt = time.time()-t0
        time.sleep(max(self.n_samples/273000 - dt,0))
        self.newData.emit(self.data)


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
        print('binned data : shape {}'.format(self.data_output.shape))
        self.data_output = binned_signal / normarray

    def bin_data(self):
        self.data_output = bin_dc(self.data_input, self.bins)


class Processor(Worker):

    def __init__(self, data, use_dark_control=True):

        super().__init__()
        self.shaker_positions = data[0]
        self.signal = data[1]
        self.dark_control = data[2]


        self.use_dark_control = use_dark_control

        step = 0.000152587890625
        minpos = self.shaker_positions.min()
        min_t = (minpos/step)*.05 # consider 0.05 ps step size from shaker digitalized signal
        maxpos = self.shaker_positions.max()
        max_t = (maxpos/step)*.05

        n_points = int((maxpos - minpos) / step)

        self.position_bins = np.array((self.shaker_positions - minpos) / step, dtype=int)
        self.result = np.zeros(n_points + 1, dtype=np.float64)
        self.normamlization_array = np.zeros(n_points + 1, dtype=np.float64)

        self.output_array = np.zeros((2,n_points+1))
        self.output_array[0] = np.linspace(min_t, max_t, n_points+1)

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()

        try:
            if self.use_dark_control:
                self.project_dc()
            else:
                self.project()
                print(self.output_array.shape,self.result.shape,self.normamlization_array.shape)
            self.output_array[1] = self.result / self.normamlization_array

            self.newData.emit(self.output_array)
            print(' - Projected {} points to a {} pts array, with {} nans in : {:.2f} ms'.format(
                len(self.signal),len(self.result), len(self.result) - len(self.output_array[1][np.isfinite(self.output_array[1])]), 1000 * (time.time() - t0)))
        except Exception as e:
            print(self.output_array.shape, self.result.shape, self.normamlization_array.shape)

            print('failed to project data with shape {} to shape {}.')
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
    ready_for_data = QtCore.pyqtSignal(int)
    def __init__(self,id):

        super().__init__()
        self.id = id

    @QtCore.pyqtSlot()
    def work(self,data,use_dark_control=True):
        t0 = time.time()
        shaker_positions = data[0]
        signal = data[1]
        dark_control = data[2]


        step = 0.000152587890625
        minpos = shaker_positions.min()
        min_t = (minpos/step)*.05 # consider 0.05 ps step size from shaker digitalized signal
        maxpos = shaker_positions.max()
        max_t = (maxpos/step)*.05

        n_points = int((maxpos - minpos) / step)

        position_bins = np.array((shaker_positions - minpos) / step, dtype=int)
        result = np.zeros(n_points + 1, dtype=np.float64)
        normamlization_array = np.zeros(n_points + 1, dtype=np.float64)

        output_array = np.zeros((2,n_points+1))
        output_array[0] = np.linspace(min_t, max_t, n_points+1)

        try:
            if use_dark_control:
                for val, pos in zip(self.signal, self.position_bins):
                    self.result[pos] += val
                    self.normamlization_array[pos] += 1.

            else:
                for val, pos in zip(self.signal, self.position_bins):
                    self.result[pos] += val
                    self.normamlization_array[pos] += 1.

                print(output_array.shape,result.shape,normamlization_array.shape)
            output_array[1] = result / normamlization_array

            self.newData.emit(output_array)
            print(' - Projected {} points to a {} pts array, with {} nans in : {:.2f} ms'.format(
                len(signal),len(result), len(result) - len(output_array[1][np.isfinite(output_array[1])]), 1000 * (time.time() - t0)))
        except Exception as e:
            print(output_array.shape, result.shape, normamlization_array.shape)

            print('failed to project data with shape {} to shape {}.')
            self.error.emit(e)
        time.sleep(0.002)

        self.isReady.emit(self.id)


class DataManager(Worker):
    """
    This should get data from streamer, through signals in main gui and assign processors to analyze it.
    TODO: implement this class in mainwinow.


    """
    def __init__(self, buffer_size=30000):
        super().__init__()
        self.stream_queue = mp.Queue()

        self.res_from_previous= np.zeros((3,0))
        self.buffer_size = buffer_size

        self.timer = QtCore.QTimer(1)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()


        self.processors = []
        self.threads = []
        self.processor_ready = []
        for i in range(os.cpu_count()-2):

            self.processors.append(Processor_multi(i))
            self.threads.append(Thread())
            self.processor_ready.append(False)
            self.processors[i].newData[np.ndarray].connect(self.on_processor_data)
            self.processors[i].error.connect(self.error.emit)
            self.processors[i].isReady.connect(self.set_processor_ready)
            self.processors[i].finished.connect(self.on_processor_finished)

            self.processors[i].moveToThread(self.thread[i])
            self.thread[i].started.connect(self.processors[i].work)
            self.thread[i].start()

    @QtCore.pyqtSlot()
    def set_processor_ready(self,id):
        self.processor_ready[id] = True

    @QtCore.pyqtSlot()
    def add_data_to_queue(self,streamer_data):
        """divide data in smaller chunks, for faster data processing."""
        if self.rest_from_previous.shape[1] > 0:
            streamer_data = np.append(self.rest_from_previous,streamer_data,axis=1)

        n_chunks = streamer_data.shape[1]//self.buffer_size

        chunks = np.array_split(streamer_data,n_chunks, axis=1)
        if chunks[-1].shape[1] < self.buffer_size:
            self.rest_from_previous = chunks.pop(-1)
        for chunk in chunks:
            self.stream_queue.put(chunk)

    @QtCore.pyqtSlot()
    def on_timer(self):
        for processor,ready in zip(self.processors,self.processor_ready):
            if ready:
                processor.work(self.stream_queue.get(),use_dark_control=True)

    @QtCore.pyqtSlot()
    def on_processor_data(self,processed_data):
        self.newData.emit(processed_data)

    def on_thread_error(self,e):
        self.error.emit(e)






class Fitter(Worker):

    def __init__(self, x,y,model='sech2',guess=None):
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
            self.guess = (A,x0,fwhm,offset)
        else:
            self.guess = guess

    @QtCore.pyqtSlot()
    def work(self):
        t0 = time.time()
        try:
            self.popt, pcov = curve_fit(self.model, self.x[np.isfinite(self.y)], self.y[np.isfinite(self.y)], p0=self.guess)
            self.newData.emit(self.popt)

        except Exception as e:
            self.error.emit(e)
            self.newData.emit(self.popt)

        self.finished.emit()



def main():
    pass


if __name__ == '__main__':
    pass
