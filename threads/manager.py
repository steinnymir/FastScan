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

import numpy as np
import xarray as xr
from PyQt5 import QtCore

from threads.core import Thread, Worker
from threads.processor import Processor_multi
from threads.streamer import Streamer


def main():
    pass


if __name__ == '__main__':
    main()


class ThreadManager(Worker):
    """
    This class manages the streamer processor and fitter workers for the fast
    scan data acquisition and processing.


    """
    newStreamerData = QtCore.pyqtSignal(np.ndarray)
    newProcessedData = QtCore.pyqtSignal(dict)
    acquisitionStopped = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, processor_buffer=30000, streamer_buffer=90000):
        super().__init__()

        self.logger = logging.getLogger('{}.ThreadManager'.format(__name__))

        self.logger.info('Created Thread Manager')

        self._SIMULATE = False
        self._darkcontrol = False
        self.stream_queue = mp.Queue()  # Queue where to store unprocessed streamer data
        self.processor_queue = mp.Queue()
        self.data_dict = {} #dict containing data to be plotted
        # self.dataset = xr.Dataset()  # dataset containing averages, fits etc...
        # self.da_all = None # will be DataArray containing all scans

        self.res_from_previous = np.zeros((3, 0))
        self.__processor_buffer_size = processor_buffer
        self.__streamer_buffer_size = streamer_buffer

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1000./60)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()

        self.processed_averages = None  # container for the xarray dataarray of all averages

        self.create_processors()
        self.create_streamer()

    # %%========== Properties ===========
    def toggle_simulation(self, bool):
        self.stop_streamer()
        self._SIMULATE = bool
        self.create_streamer()

    def toggle_darkcontrol(self, bool):
        self._darkcontrol = bool

    @property
    def processor_buffer_size(self):
        return self.__processor_buffer_size

    @processor_buffer_size.setter
    def processor_buffer_size(self, buffer_size):
        assert 0 < buffer_size < 1000000
        if not self.streamer_thread.isRunnung():

            self.__processor_buffer_size = buffer_size
        else:
            self.logger.warning('Cannot change buffer size while streamer is running.')

    @property
    def streamer_buffer_size(self):
        return self.__streamer_buffer_size

    @streamer_buffer_size.setter
    def streamer_buffer_size(self, buffer_size):
        assert 0 < buffer_size < 1000000
        self.__streamer_buffer_size = buffer_size

    def set_streamer_buffer(self, val):
        self.streamer_buffer_size = val

    def set_processor_buffer(self, val):
        self.processor_buffer_size = val

    # %%========== Methods ===========
    @QtCore.pyqtSlot()
    def on_timer(self):

        self.process_available_stream_data()
        if not self.processor_queue.empty():  # add a processed dataarray to the collection dataset
            da = self.processor_queue.get()
            try:
                self.data_dict['all'] = xr.concat([self.data_dict['all'], da], 'avg')
                self.calculate_average(100)
            except KeyError:
                self.data_dict['all'] = da

            self.newProcessedData.emit(self.data_dict)
            self.logger.debug('emitting processed dataset')

    def calculate_average(self,n):
        if 'avg' in self.data_dict['all'].dims:
            self.data_dict['average'] = self.data_dict['all'][-n:].mean('avg')

    def process_available_stream_data(self):
        """ picks the first ready processor and passes a chunk of data."""
        for processor, ready in zip(self.processors, self.processor_ready):
            if ready:
                if not self.stream_queue.empty():
                    self.logger.debug('processing data with processor {}'.format(processor.id))
                    processor.work(self.stream_queue.get(), use_dark_control=self._darkcontrol)

    def create_streamer(self):

        self.streamer_thread = Thread()
        self.streamer_thread.stopped.connect(self.acquisitionStopped.emit)

        self.streamer = Streamer(self.streamer_buffer_size, simulate=self._SIMULATE)
        self.streamer.newData[np.ndarray].connect(self.on_streamer_data)
        self.streamer.error.connect(self.error.emit)
        self.streamer.finished.connect(self.on_streamer_finished)

        self.streamer.moveToThread(self.streamer_thread)
        self.streamer_thread.started.connect(self.streamer.start_acquisition)

    @QtCore.pyqtSlot()
    def start_streamer(self):
        self.streamer_thread.start()

    @QtCore.pyqtSlot()
    def stop_streamer(self):
        self.streamer.stop_acquisition()
        self.streamer_thread.exit()
        # self.streamer_thread.stop()

    @QtCore.pyqtSlot(np.ndarray)
    def on_streamer_data(self, streamer_data):
        """divide data in smaller chunks, for faster data processing.

        Splits data from streamer in chunks whose size is defined by
        self.processor_buffer_size. If the last chunk is smaller than this, it keeps it
        and will append the next data recieved to it.

        """
        try:
            if self.rest_from_previous.shape[1] > 0:
                streamer_data = np.append(self.rest_from_previous, streamer_data, axis=1)
        except:
            pass
        n_chunks = streamer_data.shape[1] // self.processor_buffer_size

        chunks = np.array_split(streamer_data, n_chunks, axis=1)
        if chunks[-1].shape[1] < self.processor_buffer_size:
            self.rest_from_previous = chunks.pop(-1)
        for chunk in chunks:
            self.stream_queue.put(chunk)
        self.logger.debug('added {} chunks to queue'.format(n_chunks))
        self.newStreamerData.emit(streamer_data)

    def on_streamer_finished(self):
        self.logger.debug('streamer finished working')

    def create_processors(self, n_processors=os.cpu_count() - 2):
        """ create n_processors number of threads for processing streamer data"""
        self.processors = []
        self.processor_threads = []
        self.processor_ready = []
        for i in range(n_processors):
            self.processors.append(Processor_multi(i))
            self.processor_threads.append(Thread())
            self.processor_ready.append(False)
            self.processors[i].newData[np.ndarray].connect(self.on_processor_data)
            self.processors[i].error.connect(self.error.emit)
            self.processors[i].isReady.connect(self.set_processor_ready)
            self.processors[i].finished.connect(self.on_processor_finished)

            self.processors[i].moveToThread(self.processor_threads[i])
            self.processor_threads[i].started.connect(self.processors[i].initialize)
            self.processor_threads[i].start()

    @QtCore.pyqtSlot(int)
    def set_processor_ready(self, id):
        self.processor_ready[id] = True

    @QtCore.pyqtSlot(xr.DataArray)
    def on_processor_data(self, processed_dataarray):
        """ called when new processed data is available

        This emits data to the main window, so it can be plotted..."""
        self.processor_queue.put(processed_dataarray)
        # self.newProcessedData.emit(processed_data)

    @QtCore.pyqtSlot()
    def on_processor_finished(self):
        self.logger.debug('Processor finished working')

    def reset_data(self):
        self.dataset = xr.Dataset()

    def close(self):
        self.stop_streamer()
        for thread in self.processor_threads:
            thread.exit()
