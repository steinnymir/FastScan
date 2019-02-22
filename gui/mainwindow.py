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
import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, \
    QGroupBox, QSpinBox
from pyqtgraph.Qt import QtCore, QtGui

from multiprocessing import Pool

from gui.threads import Streamer, Thread, Binner, Projector
from utilities.data import bin_dc, bin_dc_multi


class FastScanMainWindow(QMainWindow):
    _SIMULATE = False

    def __init__(self):
        super(FastScanMainWindow, self).__init__()
        self.setWindowTitle('Fast Scan')
        self.setGeometry(100, 50, 1152, 768)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('ready')

        # set the cool dark theme and other plotting settings
        # self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        pg.setConfigOption('background', .1)
        pg.setConfigOption('foreground', .9)
        pg.setConfigOptions(antialias=True)
        self.setupUi()

        #########################
        #   define variables    #
        #########################
        # self.main_clock = QTimer()
        # self.set_clock_frequency(2)
        # self.main_clock.timeout.connect(self.on_timer)
        # self.main_clock_running = False

        self.laser_trigger_frequency = 273000
        self.shaker_frequency = 10
        self.n_periods = 10

        self.n_samples = int((self.laser_trigger_frequency / self.shaker_frequency) * self.n_periods)

        self.n_points = 1000
        self.shaker_amplitude = 100*1e-12
        self.time_axis = self.make_time_axis()
        self.signal_averages = []
        self.current_average = np.zeros_like(self.time_axis)

        self.dark_control = False
        self.pp_method = Projector # project or bin: these are the accepted methods

        self.unprocessed_data = np.zeros((3,0))

        self._processing = False

    def setupUi(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)

        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)

        control_widget = QWidget()
        central_layout.addWidget(control_widget)
        control_layout = QVBoxLayout()
        control_widget.setLayout(control_layout)

        visual_widget = QWidget()
        central_layout.addWidget(visual_widget)
        visual_layout = QGridLayout()
        visual_widget.setLayout(visual_layout)

        self.top_plot_area = pg.PlotWidget(name='top_plot')
        visual_layout.addWidget(self.top_plot_area, 0, 0, 1,1)
        self.top_plot_area.showAxis('top',True)
        self.top_plot_area.showAxis('right',True)
        self.top_plot_area.showGrid(True,True,.2)

        self.top_plot = self.top_plot_area.plot()
        self.top_plot.setPen(pg.mkPen(255, 255, 255))
        self.top_plot_area.setLabel('left', 'Value', units='V')
        self.top_plot_area.setLabel('bottom', 'Time', units='s')

        self.bot_plot_area = pg.PlotWidget(name='top_plot')
        visual_layout.addWidget(self.bot_plot_area, 1, 0, 1, 1)
        self.bot_plot_area.showAxis('top',True)
        self.bot_plot_area.showAxis('right',True)
        self.bot_plot_area.showGrid(True,True,.2)

        self.bot_plot = self.bot_plot_area.plot()
        self.bot_plot.setPen(pg.mkPen(100, 100, 100))
        self.bot_plot_avg = self.bot_plot_area.plot()
        self.bot_plot_avg.setPen(pg.mkPen(100, 255, 100))
        self.bot_plot_area.setLabel('left', 'Value', units='V')
        self.bot_plot_area.setLabel('bottom', 'Time', units='s')

        box_cont = QGroupBox('continuous acquisition')
        control_layout.addWidget(box_cont)
        box_cont_layout = QGridLayout()
        box_cont.setLayout(box_cont_layout)

        self.start_button = QPushButton('start')
        box_cont_layout.addWidget(self.start_button, 0, 0, 1, 1)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton('stop')
        box_cont_layout.addWidget(self.stop_button, 0, 1, 1, 1)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)


        control_layout.addItem(verticalSpacer)

    def make_time_axis(self):
        self.time_axis = np.linspace(-self.shaker_amplitude/2,self.shaker_amplitude/2,self.n_points)
        return self.time_axis

    def start_acquisition(self):
        self.start_button.setEnabled(False)

        self.status_bar.showMessage('initializing acquisition')
        self.streamer_thread = Thread()
        self.streamer_thread.stopped.connect(self.kill_streamer_thread)
        self.streamer = Streamer(self.n_samples,simulate=self._SIMULATE)
        self.streamer.newData[np.ndarray].connect(self.on_streamer_data)
        self.streamer.error.connect(self.raise_thread_error)
        self.streamer.finished.connect(self.on_streamer_finished)
        self.streamer.moveToThread(self.streamer_thread)
        self.streamer_thread.started.connect(self.streamer.start_acquisition)
        self.streamer_thread.start()
        self.status_bar.showMessage('Acquisition running')
        self.stop_button.setEnabled(True)

    def stop_acquisition(self):
        self.status_bar.showMessage('Stopping aquisition')
        self.stop_button.setEnabled(False)
        print('attempting to stop thread')
        self.streamer.stop_acquisition()
        self.streamer_thread.exit()
        self.status_bar.showMessage('Ready')
        self.start_button.setEnabled(True)

    @QtCore.pyqtSlot(np.ndarray)
    def on_streamer_data(self, data):
        print('stream data recieved')
        self.draw_top_plot(np.linspace(0, len(data[0])/self.laser_trigger_frequency, len(data[0])), data[0])
        self.unprocessed_data = np.append(self.unprocessed_data, data, axis=1)
        print(self.unprocessed_data.shape,data.shape)
        self.process_data(data)
        self.unprocessed_data = np.zeros((3,0))


    def on_streamer_finished(self):
        print('streamer finished signal recieved')

    def kill_streamer_thread(self):
        print('streamer Thread finished, deleting instance')
        self.streamer_thread = None
        self.streamer = None

    def process_data(self,data):
        self._processing = True

        self.processor_thread = Thread()
        self.processor_thread.stopped.connect(self.kill_processor_thread)
        self.processor = self.pp_method(data,self.n_points,dark_control=self.dark_control)
        self.processor.newData[np.ndarray].connect(self.on_processor_data)
        self.processor.error.connect(self.raise_thread_error)
        self.streamer.finished.connect(self.on_processor_finished)
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.started.connect(self.processor.work)
        self.processor_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def on_processor_data(self, data):
        self._processing = False
        self.processor_thread.exit()
        self.signal_averages.append(data)
        self.draw_bot_plot()


    def on_processor_finished(self):
        print('Processor finished signal recieved')

    def kill_processor_thread(self):
        print('processor_thread Thread finished, deleting instance')
        self.binner_thread = None
        self.processor = None
    #
    def raise_thread_error(self, e):
        print('---Error---\n{}'.format(e))

    def draw_top_plot(self, xd, yd):
        self.top_plot.setData(x=xd, y=yd)

    def draw_bot_plot(self):
        y = self.signal_averages[-1]
        yavg = np.array(self.signal_averages).mean(axis=0)
        self.bot_plot.setData(x=self.time_axis[2:-2], y=y[2:-2])
        self.bot_plot_avg.setData(x=self.time_axis[2:-2], y=yavg[2:-2])



    def closeEvent(self, event):
        # geometry = self.saveGeometry()
        # self.qsettings.setValue('geometry', geometry)
        super(FastScanMainWindow, self).closeEvent(event)
        print('quitted properly')


def main():
    pass


if __name__ == '__main__':
    main()
