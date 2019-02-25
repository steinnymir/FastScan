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
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, \
    QGroupBox,QLabel, QLineEdit
from PyQt5.QtCore import QTimer
from pyqtgraph.Qt import QtCore, QtGui

import h5py

import time
import qdarkstyle

from gui.threads import Streamer, Thread, Projector
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
        pg.setConfigOption('background', .1)
        pg.setConfigOption('foreground', .9)
        pg.setConfigOptions(antialias=True)


        #########################
        #   define variables    #
        #########################






        self.settings = {'laser_trigger_frequency':273000,
                         'shaker_frequency':10,
                         'n_samples':100000,
                         'shaker_amplitude':10,
                         'n_plot_points':15000
                         }

        self.data = {'processed':None,
                     'unprocessed': np.zeros((3, 0)),
                     'time_axis': None,
                     'last_trace': None,
                     'all_traces': None,
                     }


        self._processing_tick = None
        self._streamer_tick = None

        self.pp_method = Projector  # project or bin: these are the accepted methods
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

        box_cont = QGroupBox('Acquisition')
        layout.addWidget(box_cont)
        box_cont_layout = QGridLayout()
        box_cont.setLayout(box_cont_layout)

        self.start_button = QPushButton('start')
        box_cont_layout.addWidget(self.start_button, 0, 0, 1, 1)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton('stop')
        box_cont_layout.addWidget(self.stop_button, 0, 1, 1, 1)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)

        self.reset_button = QPushButton('reset')
        box_cont_layout.addWidget(self.reset_button, 1, 1, 1, 1)
        self.reset_button.clicked.connect(self.reset_data)
        self.reset_button.setEnabled(True)

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


        self.save_box = QGroupBox('Save')
        savebox_layout = QHBoxLayout()
        self.save_box.setLayout(savebox_layout)

        self.save_name_ledit = QLineEdit('D:/data/fastscan/test01')
        savebox_layout.addWidget(self.save_name_ledit)

        self.save_data_button = QPushButton('Save')
        savebox_layout.addWidget(self.save_data_button)
        self.save_data_button.clicked.connect(self.save_data)

        layout.addItem(self.__verticalSpacer)
        layout.addWidget(self.save_box)
        return widget

    def save_data(self):
        filename = self.save_name_ledit.text()

        dir = '\\'.join(filename.split('/')[:-1])
        name = filename.split('/')[-1]
        if not os.path.isdir(dir):
            os.mkdir(dir)

        with h5py.File(os.path.join(dir,name+".h5"), "w") as f:
            data_grp = f.create_group('data')
            settings_grp = f.create_group('settings')

            for key,val in self.data.items():
                if val is not None:
                    print('saving {},{}'.format(key,val))
                    data_grp.create_dataset(key,data=val)
            for key,val in self.settings.items():
                if val is not None:
                    settings_grp.create_dataset(key,data=val)
                    print('saving {},{}'.format(key,val))


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

    @QtCore.pyqtSlot()
    def reset_data(self):

        self.data = {'raw':None,
                     'processed':None,
                     'unprocessed': np.zeros((3, 0)),
                     'time_axis': None,
                     'last_trace': None,
                     'all_traces': None,
                     }
    def make_time_axis(self):
        amp = self.settings['shaker_amplitude']
        n_pts = self.settings['n_plot_points']
        self.data['time_axis'] = np.linspace(-amp,amp,self.settings['n_plot_points'])
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
        self.status_bar.showMessage('Stopping aquisition')
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
        self.draw_raw_signal_plot(np.linspace(0, len(data[0]) / self.settings['laser_trigger_frequency'], len(data[0])), data[0])

    def on_streamer_finished(self):
        print('streamer finished signal recieved')

    def kill_streamer_thread(self):
        print('streamer Thread finished, deleting instance')
        self.streamer_thread = None
        self.streamer = None

    def on_main_clock(self):
        if not self._processing and self.data['unprocessed'].shape[1]>0:
            t=time.time()
            if self._processing_tick is not None:
                dt = 1./(t-self._processing_tick)
                if dt>1:
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
        self.processor_thread = Thread()
        self.processor_thread.stopped.connect(self.kill_processor_thread)
        self.processor = self.pp_method(data, self.settings['n_plot_points'], dark_control=self.dark_control)
        self.processor.newData[np.ndarray].connect(self.on_processor_data)
        self.processor.error.connect(self.raise_thread_error)
        self.processor.finished.connect(self.on_processor_finished)
        self.processor.moveToThread(self.processor_thread)
        self.processor_thread.started.connect(self.processor.work)
        self.processor_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def on_processor_data(self, data):
        print('processed data into {}'.format(data.shape))
        self._processing = False
        self.processor_thread.exit()
        # self.data['last_trace'] = data
        if self.data['all_traces'] is None:
            self.data['all_traces'] = []
        self.data['all_traces'].append(data)
        self.draw_main_plot()

    def on_processor_finished(self):
        print('Processor finished signal recieved')

    def kill_processor_thread(self):
        print('processor_thread Thread finished, deleting instance')
        self.binner_thread = None
        self.processor = None

    def raise_thread_error(self, e):
        print('---Error---\n{}'.format(e))

    def draw_raw_signal_plot(self, xd, yd):
        self.raw_data_plot.setData(x=xd, y=yd)

    def draw_main_plot(self):
        if self.data['time_axis']is None:
            self.make_time_axis()
        x = self.data['time_axis']
        y = self.data['all_traces'][-1]
        print(len(x))
        print(len(y))
        yavg = np.array(self.data['all_traces']).mean(axis=0)
        self.plot_back_line.setData(x=x[2:-2], y=y[2:-2])
        self.plot_front_line.setData(x=x[2:-2], y=yavg[2:-2])

    def closeEvent(self, event):
        # geometry = self.saveGeometry()
        # self.qsettings.setValue('geometry', geometry)
        super(FastScanMainWindow, self).closeEvent(event)
        print('quitted properly')


def main():
    pass


if __name__ == '__main__':
    main()
