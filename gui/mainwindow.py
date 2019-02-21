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

from gui.threads import Streamer, Thread, Binner
from utilities.data import bin_dc, bin_dc_multi


class FastScanMainWindow(QMainWindow):
    _SIMULATE = True

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

        self.bins = None
        self.n_bins = 1000
        self.bin_cutoff = .02

        self.unprocessed_data = np.zeros((3,0))

        self._binning = False

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
        self.top_plot = self.top_plot_area.plot()
        self.top_plot.setPen(pg.mkPen(255, 255, 255))
        self.top_plot_area.setLabel('left', 'Value', units='V')
        self.top_plot_area.setLabel('bottom', 'Time', units='s')

        self.bot_plot_area = pg.PlotWidget(name='top_plot')
        visual_layout.addWidget(self.bot_plot_area, 1, 0, 1, 1)
        self.bot_plot = self.bot_plot_area.plot()
        self.bot_plot.setPen(pg.mkPen(255, 255, 255))
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


    def make_bins(self, data, n_bins=1000, cutoff=.02):
        dmax, dmin = data[0].max(), data[0].min()
        amp = dmax - dmin
        dmax -= amp * cutoff
        dmin += amp * cutoff
        self.bins = np.linspace(dmin, dmax, n_bins)

    # def set_clock_frequency(self, frequency=2):
    #     self.main_clock.setInterval(1. / frequency)
    #
    # def start_stop_timer(self):
    #     if self.main_clock_running:
    #         self.main_clock.stop()
    #         self.main_clock_running = False
    #         self.status_bar.showMessage('Main Clock Started')
    #     else:
    #         self.main_clock.start()
    #         self.main_clock_running = True
    #         self.status_bar.showMessage('Main Clock Stopped')
    #
    # def on_timer(self):
    #     shaker_position, signal, n = self.measure()
    #     self.draw_top_plot(n, shaker_position)

    def start_acquisition(self):
        self.start_button.setEnabled(False)

        self.status_bar.showMessage('initializing acquisition')
        self.streamer_thread = Thread()
        self.streamer_thread.stopped.connect(self.kill_streamer_thread)
        self.streamer = Streamer(self.n_samples)
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
        self.draw_top_plot(np.linspace(0, len(data[0]) - 1, len(data[0])), data[1])

        self.unprocessed_data = np.append(self.unprocessed_data, data, axis=1)
        print(self.unprocessed_data.shape,data.shape)
        if not self._binning:
            print('binning data of shape {}'.format(self.unprocessed_data.shape))
            self.bin_data(self.unprocessed_data)
            self.unprocessed_data = np.zeros((3,0))

    def on_streamer_finished(self):
        print('streamer finished signal recieved')

    def kill_streamer_thread(self):
        print('streamer Thread finished, deleting instance')
        self.streamer_thread = None
        self.streamer = None

    def bin_data(self, data):
        self._binning = True
        if self.bins is None:
            self.make_bins(data)
        self.binner_thread = Thread()
        self.streamer_thread.stopped.connect(self.kill_binner_thread)
        self.binner = Binner(data,self.bins)
        self.binner.newData[np.ndarray].connect(self.on_binner_data)
        self.binner.error.connect(self.raise_thread_error)
        self.streamer.finished.connect(self.on_binner_finished)
        self.binner.moveToThread(self.binner_thread)
        self.binner_thread.started.connect(self.binner.work)
        self.binner_thread.start()


    @QtCore.pyqtSlot(np.ndarray)
    def on_binner_data(self, data):
        self._binning = False
        self.binner_thread.exit()
        self.draw_bot_plot(self.bins, data)

    def on_binner_finished(self):
        print('streamer finished signal recieved')

    def kill_binner_thread(self):
        print('binner Thread finished, deleting instance')
        self.binner_thread = None
        self.binner = None
    #
    def raise_thread_error(self, e):
        print('Rising error from thread')
        print(e)

    def draw_top_plot(self, xd, yd):
        self.top_plot.setData(x=xd, y=yd)

    def draw_bot_plot(self,xd,yd):
        self.bot_plot.setData(x=xd, y=yd)

    def closeEvent(self, event):
        # geometry = self.saveGeometry()
        # self.qsettings.setValue('geometry', geometry)
        super(FastScanMainWindow, self).closeEvent(event)
        print('quitted properly')


def main():
    pass


if __name__ == '__main__':
    main()
