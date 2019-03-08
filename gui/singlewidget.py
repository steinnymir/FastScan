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
import time

import pandas as pd

import h5py
import numpy as np
import pyqtgraph as pg
import qdarkstyle
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, \
    QGroupBox, QLabel, QLineEdit
from pyqtgraph.Qt import QtCore, QtGui

from threads import Processor, Streamer
from threads.core import Thread
from threads.fitting import Fitter

from utilities.data import fit_peak
from utilities.math import gaussian_fwhm, sech2_fwhm
from utilities.qt import SpinBox, labeled_qitem


class FastScanMainWindow(QMainWindow):
    _SIMULATE = True

    def __init__(self):
        super(FastScanMainWindow, self).__init__()
        self.setWindowTitle('Fast Scan')
        self.setGeometry(100, 50, 1152, 768)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('ready')
        # set the cool dark theme and other plotting settings
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        pg.setConfigOption('background', (25, 35, 45))
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOptions(antialias=True)

        #########################
        #   define variables    #
        #########################

        self.settings = {'laser_trigger_frequency': 273000,
                         'shaker_frequency': 10,
                         'n_samples': 60000,
                         'shaker_amplitude': 10,
                         'n_plot_points': 15000
                         }

        self.data = {'processed': None,
                     'unprocessed': np.zeros((3, 0)),
                     'time_axis': None,
                     'last_trace': None,
                     'all_traces': None,
                     'df_traces': None,
                     'df_averages': None
                     }

        self._processing_tick = None
        self._streamer_tick = None

        self.peak_fit_parameters = None
        self.peak_fit_data = None

        # self.processing_method = 'project'  # project or bin: these are the accepted methods
        self._processing = False

        self.main_clock = QTimer()
        self.main_clock.setInterval(1. / 60)
        self.main_clock.timeout.connect(self.on_main_clock)
        self.main_clock.start()
        self.setupUi()

    def setupUi(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)

        self.__verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)

        control_widget = self.make_controlwidget()
        visual_widget = self.make_visualwidget()

        main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(visual_widget)
        # main_splitter.setStretchFactor(0, 5)

        central_layout.addWidget(main_splitter)

    def make_controlwidget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # ----------------------------------------------------------------------
        # Settings Box
        # ----------------------------------------------------------------------

        acquisition_box = QGroupBox('Acquisition')
        layout.addWidget(acquisition_box)
        acquisition_box_layout = QGridLayout()
        acquisition_box.setLayout(acquisition_box_layout)

        self.start_button = QPushButton('start')
        acquisition_box_layout.addWidget(self.start_button, 0, 0, 1, 1)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton('stop')
        acquisition_box_layout.addWidget(self.stop_button, 0, 1, 1, 1)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)

        self.reset_button = QPushButton('reset')
        acquisition_box_layout.addWidget(self.reset_button, 1, 1, 1, 1)
        self.reset_button.clicked.connect(self.reset_data)
        self.reset_button.setEnabled(True)

        # ----------------------------------------------------------------------
        # Settings Box
        # ----------------------------------------------------------------------
        settings_box = QGroupBox('settings')
        layout.addWidget(settings_box)
        settings_box_layout = QGridLayout()
        settings_box.setLayout(settings_box_layout)

        settings_items = []

        self.spinbox_laser_trigger_frequency = SpinBox(
            name='Trigger Frequency', layout_list=settings_items,
            type=int, value=self.settings['laser_trigger_frequency'], max='max', step=1, suffix='Hz')
        self.spinbox_laser_trigger_frequency.valueChanged.connect(self.set_laser_trigger_frequency)

        self.spinbox_shaker_frequency = SpinBox(
            name='Shaker Frequency', layout_list=settings_items, type=float,
            value=self.settings['shaker_frequency'], max=20, step=.01, suffix='Hz')
        self.spinbox_shaker_frequency.valueChanged.connect(self.set_shaker_frequency)

        self.spinbox_n_samples = SpinBox(
            name='samples', layout_list=settings_items,
            type=int, value=self.settings['n_samples'], step=1, max='max')
        self.spinbox_n_samples.valueChanged.connect(self.set_n_samples)

        self.spinbox_shaker_amplitude = SpinBox(
            name='Shaker Amplitude', layout_list=settings_items,
            type=float, value=self.settings['shaker_amplitude'], step=.01, suffix='ps')
        self.spinbox_shaker_amplitude.valueChanged.connect(self.set_shaker_amplitude)

        self.spinbox_n_plot_points = SpinBox(
            name='Plot points', layout_list=settings_items,
            type=int, value=self.settings['n_plot_points'], step=1, max='max')
        self.spinbox_n_plot_points.valueChanged.connect(self.set_n_plot_points)

        for item in settings_items:
            settings_box_layout.addWidget(labeled_qitem(*item))
        self.label_processor_fps = QLabel('FPS: 0')
        settings_box_layout.addWidget(self.label_processor_fps)
        self.label_streamer_fps = QLabel('FPS: 0')
        settings_box_layout.addWidget(self.label_streamer_fps)
        self.radio_dark_control = QRadioButton('Dark Control')
        self.radio_dark_control.setChecked(True)
        settings_box_layout.addWidget(self.radio_dark_control)

        # ----------------------------------------------------------------------
        # Autocorrelation Box
        # ----------------------------------------------------------------------

        autocorrelation_box = QGroupBox('Autocorrelation')
        autocorrelation_box_layout = QGridLayout()
        autocorrelation_box.setLayout(autocorrelation_box_layout)

        self.fit_off_checkbox = QRadioButton('Off')
        autocorrelation_box_layout.addWidget(self.fit_off_checkbox, 0, 0, 1, 1)
        self.fit_gauss_checkbox = QRadioButton('Gaussian')
        autocorrelation_box_layout.addWidget(self.fit_gauss_checkbox, 0, 1, 1, 1)
        self.fit_sech2_checkbox = QRadioButton('Sech2')
        autocorrelation_box_layout.addWidget(self.fit_sech2_checkbox, 0, 2, 1, 1)

        self.fit_off_checkbox.setChecked(True)

        font = QFont()
        font.setBold(True)
        font.setPointSize(16)
        self.fit_report_label = QLabel('Fit parameters:\n')
        autocorrelation_box_layout.addWidget(self.fit_report_label, 2, 0)
        self.pulse_duration_label = QLabel('0 fs')

        self.pulse_duration_label.setFont(font)

        autocorrelation_box_layout.addWidget(QLabel('Pulse duration:'), 3, 0)
        autocorrelation_box_layout.addWidget(self.pulse_duration_label, 3, 1)

        layout.addWidget(autocorrelation_box)
        # layout.addItem(self.__verticalSpacer)

        # ----------------------------------------------------------------------
        # Save Box
        # ----------------------------------------------------------------------

        save_box = QGroupBox('Save')
        savebox_layout = QHBoxLayout()
        save_box.setLayout(savebox_layout)
        self.save_name_ledit = QLineEdit('D:/data/fastscan/test01')
        savebox_layout.addWidget(self.save_name_ledit)
        self.save_data_button = QPushButton('Save')
        savebox_layout.addWidget(self.save_data_button)
        self.save_data_button.clicked.connect(self.save_data)
        layout.addWidget(save_box)

        return widget

    def make_visualwidget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.main_plot_widget = pg.PlotWidget(name='raw_data_plot')
        self.setup_plot_widget(self.main_plot_widget, title='Signal')
        self.main_plot_widget.setMinimumHeight(450)

        self.plot_back_line = self.main_plot_widget.plot()
        self.plot_back_line.setPen(pg.mkPen(100, 100, 100))
        self.plot_front_line = self.main_plot_widget.plot()
        self.plot_front_line.setPen(pg.mkPen(100, 255, 100))
        self.plot_fit_line = self.main_plot_widget.plot()
        self.plot_fit_line.setPen(pg.mkPen(247, 211, 7))

        self.secondary_plot_widget = pg.PlotWidget(name='raw_data_plot')
        self.setup_plot_widget(self.secondary_plot_widget, title='raw data stream')

        self.raw_data_plot = self.secondary_plot_widget.plot()
        self.raw_data_plot.setPen(pg.mkPen(255, 255, 255))

        vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        vsplitter.addWidget(self.main_plot_widget)
        vsplitter.addWidget(self.secondary_plot_widget)

        # vsplitter.setStretchFactor(0, 5)
        layout.addWidget(vsplitter)

        return widget

    def setup_plot_widget(self, plot_widget, title='Plot'):
        plot_widget.showAxis('top', True)
        plot_widget.showAxis('right', True)
        plot_widget.showGrid(True, True, .2)
        plot_widget.setLabel('left', 'Value', units='V')
        plot_widget.setLabel('bottom', 'Time', units='s')
        plot_widget.setLabel('top', title)

    def save_data(self):
        filename = self.save_name_ledit.text()

        dir = '\\'.join(filename.split('/')[:-1])
        name = filename.split('/')[-1]
        if not os.path.isdir(dir):
            os.mkdir(dir)

        with h5py.File(os.path.join(dir, name + ".h5"), "w") as f:
            data_grp = f.create_group('data')
            settings_grp = f.create_group('settings')

            for key, val in self.data.items():
                if val is not None:
                    print('saving {},{}'.format(key, val))
                    data_grp.create_dataset(key, data=val)
            for key, val in self.settings.items():
                if val is not None:
                    settings_grp.create_dataset(key, data=val)
                    print('saving {},{}'.format(key, val))

    @QtCore.pyqtSlot(int)
    def set_laser_trigger_frequency(self, val):
        self.settings['laser_trigger_frequency'] = val

    @QtCore.pyqtSlot(float)
    def set_shaker_frequency(self, val):
        self.settings['shaker_frequency'] = val

    @QtCore.pyqtSlot(int)
    def set_n_samples(self, val):
        self.settings['n_samples'] = val

    @QtCore.pyqtSlot(float)
    def set_shaker_amplitude(self, val):
        self.settings['shaker_amplitude'] = val

    @QtCore.pyqtSlot(int)
    def set_n_plot_points(self, val):
        self.settings['n_plot_points'] = val

    @property
    def dark_control(self):
        return self.radio_dark_control.isChecked()

    @property
    def time_axis(self):
        return self.data['time_axis']

    @QtCore.pyqtSlot()
    def reset_data(self):

        self.data = {'raw': None,
                     'processed': None,
                     'unprocessed': np.zeros((3, 0)),
                     'time_axis': None,
                     'last_trace': None,
                     'all_traces': None,
                     }

    def make_time_axis(self):
        amp = self.settings['shaker_amplitude']
        self.data['time_axis'] = np.linspace(-amp, amp, self.settings['n_plot_points'])
        return self.data['time_axis']

    def start_acquisition(self):
        self.start_button.setEnabled(False)

        self.status_bar.showMessage('initializing acquisition')
        self.streamer_thread = Thread()
        self.streamer_thread.stopped.connect(self.kill_streamer_thread)

        self.streamer = Streamer(self.settings['n_samples'], simulate=self._SIMULATE)
        self.streamer.newData[np.ndarray].connect(self.on_streamer_data)
        self.streamer.error.connect(self.raise_thread_error)
        self.streamer.finished.connect(self.on_streamer_finished)

        self.streamer.moveToThread(self.streamer_thread)
        self.streamer_thread.started.connect(self.streamer.start_acquisition)

        self.streamer_thread.start()
        self.status_bar.showMessage('Acquisition running')
        self.stop_button.setEnabled(True)

    def stop_acquisition(self):
        self.status_bar.showMessage('Stopping acquisition')
        self.stop_button.setEnabled(False)
        print('attempting to stop thread')
        self.streamer.stop_acquisition()
        self.streamer_thread.exit()
        self.status_bar.showMessage('Ready')
        self.start_button.setEnabled(True)

    @QtCore.pyqtSlot(np.ndarray)
    def on_streamer_data(self, data):
        print('stream data recieved: {} pts'.format(data.shape))
        t = time.time()
        if self._streamer_tick is not None:
            dt = 1. / (t - self._streamer_tick)
            if dt > 1:
                self.label_streamer_fps.setText('streamer: {:.2f} frame/s'.format(dt))
            else:
                self.label_streamer_fps.setText('streamer: {:.2f} s/frame'.format(1. / dt))
        self._streamer_tick = t
        self.data['unprocessed'] = np.append(self.data['unprocessed'], data, axis=1)
        self.draw_raw_signal_plot(np.linspace(0, len(data[0]) / self.settings['laser_trigger_frequency'], len(data[0])),
                                  data[0])

    def on_streamer_finished(self):
        print('streamer finished signal recieved')

    def kill_streamer_thread(self):
        print('streamer Thread finished, deleting instance')
        self.streamer_thread = None
        self.streamer = None

    def on_main_clock(self):
        if not self._processing and self.data['unprocessed'].shape[1] > 0:
            t = time.time()
            if self._processing_tick is not None:
                dt = 1. / (t - self._processing_tick)
                if dt > 1:
                    self.label_processor_fps.setText('processor: {:.2f} frame/s'.format(dt))
                else:
                    self.label_processor_fps.setText('processor: {:.2f} s/frame'.format(1. / dt))
            self._processing_tick = t
            data = self.data['unprocessed']
            if self.data['processed'] is None:
                self.data['processed'] = data
            else:
                self.data['processed'] = np.append(self.data['processed'], data, axis=1)
            self.data['unprocessed'] = np.zeros((3, 0))
            self.process_data(data)

    def process_data(self, data):
        self._processing = True
        t = time.time()
        self.processor_thread = Thread()
        self.processor_thread.stopped.connect(self.kill_processor_thread)
        self.processor = Processor(data, use_dark_control=self.dark_control)
        self.processor.newData[np.ndarray].connect(self.on_processor_data)
        self.processor.error.connect(self.raise_thread_error)
        self.processor.finished.connect(self.on_processor_finished)
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.started.connect(self.processor.work)
        self.processor_thread.start()
        print('processor started in {:.2f} ms'.format((time.time()-t)*1000))

    @QtCore.pyqtSlot(np.ndarray)
    def on_processor_data(self, data):
        print('processed data recieved')
        self._processing = False
        self.processor_thread.exit()
        # draw data on plot
        time_axis = data[0]
        signal = data[1]
        signal_df = pd.DataFrame(data=signal, index=time_axis)

        if self.data['df_traces'] is None:
            self.data['df_traces'] = signal_df
        else:
            self.data['df_traces'] = pd.concat([self.data['df_traces'], signal_df], axis=1)

        self.draw_main_plot()

    def on_processor_finished(self):
        print('Processor finished signal recieved')

    def kill_processor_thread(self):
        print('processor_thread Thread finished, deleting instance')
        self.processor_thread = None
        self.processor = None

    def fit_data(self, time_axis, data):

        self.fitter_thread = Thread()
        self.fitter_thread.stopped.connect(self.kill_fitter_thread)
        self.fitter = Fitter(time_axis, data)
        self.fitter.newData[np.ndarray].connect(self.on_fitter_data)
        self.fitter.error.connect(self.raise_thread_error)
        self.fitter.finished.connect(self.on_fitter_finished)
        self.fitter.moveToThread(self.fitter_thread)
        self.fitter_thread.started.connect(self.fitter.work)
        self.fitter_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def on_fitter_data(self, popt):
        if popt is not None:
            self.peak_fit_parameters = popt
            if self.fit_sech2_checkbox.isChecked():
                model = sech2_fwhm
            elif self.fit_gauss_checkbox.isChecked():
                model = gaussian_fwhm
            else:
                model = None
            if model is not None:
                self.peak_fit_data = model(self.time_axis, *popt)
                self.plot_fit_line.setData(x=self.time_axis[2:-2], y=self.peak_fit_data[2:-2])
                self.fit_report_label.setText('Parameters:\nA: {} x0: {} FWHM: {}, c: {}'.format(*popt))
                self.pulse_duration_label.setText('{} fs'.format(popt[2] * 1e15))

    def on_fitter_finished(self):
        print('Fitter finished signal recieved')

    def kill_fitter_thread(self):
        print('fitter_thread Thread finished, deleting instance')
        self.fitter_thread = None
        self.fitter = None

    def raise_thread_error(self, ex):
        template = "---------- ERROR ------------\n" \
                   "An exception of type {0} occurred. Arguments:\n{1!r}" \
                   "\n-----------------------------"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

    def draw_raw_signal_plot(self, xd, yd):
        self.raw_data_plot.setData(x=xd, y=yd)

    def draw_main_plot(self):

        current = self.data['df_traces'].values[:, -1]
        x_current = self.data['df_traces'].index.values
        self.plot_back_line.setData(x=x_current[2:-2], y=current[2:-2])

        x_avg, y_avg = self.get_avg_curve(length=1000, n_averages=0)

        self.plot_front_line.setData(x=x_avg, y=y_avg)

        if self.fit_sech2_checkbox.isChecked() or self.fit_sech2_checkbox.isChecked():
            self.fit_data(x_avg, y_avg)

    def get_avg_curve(self, length=1000, n_averages=0):

        if n_averages == 0:
            df = self.data['df_traces']
        else:
            n_averages = min(len(self.data['df_traces'].columns), n_averages)
            df = self.data['df_traces'][self.data['df_traces'][:-n_averages]]

        n_plt_pts = length
        a, b = df.index.min(), df.index.max()
        bins = np.linspace(a, b, n_plt_pts + 1)
        step = bins[1] - bins[0]
        plot_bins = np.linspace(a + step / 2, b - step / 2, n_plt_pts)

        binned = df.groupby(pd.cut(df.index, bins)).mean()
        avg = binned.mean(axis=1).values
        return plot_bins, avg

    def closeEvent(self, event):
        # geometry = self.saveGeometry()
        # self.qsettings.setValue('geometry', geometry)
        super(FastScanMainWindow, self).closeEvent(event)
        print('quitted properly')


def main():
    pass


if __name__ == '__main__':
    main()
