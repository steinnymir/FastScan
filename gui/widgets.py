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

import numpy as np
import time
import pyqtgraph as pg
import qdarkstyle
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QRadioButton, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, \
    QGroupBox, QLabel, QLineEdit
from pyqtgraph.Qt import QtCore, QtGui

from gui.threads import DataManager, Thread
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
        #   define attributes    #
        #########################

        self.data_manager, self.data_manager_thread = self.make_data_manager()

        self.settings = {'streamer_buffer': 100000,
                         'processor_buffer':60000,
                         'shaker_amplitude': 10,
                         'dark_control':True
                         }


        self.fps_l = []

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
        self.visual_widget = FastScanPlotWidget()

        main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(self.visual_widget)
        # main_splitter.setStretchFactor(0, 5)

        central_layout.addWidget(main_splitter)

    def make_controlwidget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # ----------------------------------------------------------------------
        # Acquisition Box
        # ----------------------------------------------------------------------

        acquisition_box = QGroupBox('Acquisition')
        layout.addWidget(acquisition_box)
        acquisition_box_layout = QGridLayout()
        acquisition_box.setLayout(acquisition_box_layout)

        self.start_button = QPushButton('start')
        acquisition_box_layout.addWidget(self.start_button, 0, 0, 1, 1)
        self.start_button.clicked.connect(self.data_manager.start_streamer)
        self.stop_button = QPushButton('stop')
        acquisition_box_layout.addWidget(self.stop_button, 0, 1, 1, 1)
        self.stop_button.clicked.connect(self.data_manager.stop_streamer)
        # self.stop_button.setEnabled(False)

        self.reset_button = QPushButton('reset')
        acquisition_box_layout.addWidget(self.reset_button, 1, 1, 1, 1)
        self.reset_button.clicked.connect(self.reset_data)
        self.reset_button.setEnabled(True)

        self.radio_simulate = QRadioButton('Simulate')
        self.radio_simulate.setChecked(True)
        acquisition_box_layout.addWidget(self.radio_simulate,1,0,1,1)
        self.radio_simulate.clicked.connect(self.toggle_simulation_mode)
        self.toggle_simulation_mode()


        # ----------------------------------------------------------------------
        # Settings Box
        # ----------------------------------------------------------------------
        settings_box = QGroupBox('settings')
        layout.addWidget(settings_box)
        settings_box_layout = QGridLayout()
        settings_box.setLayout(settings_box_layout)

        settings_items = []

        self.spinbox_streamer_buffer = SpinBox(
            name='samples', layout_list=settings_items,
            type=int, value=self.settings['streamer_buffer'], step=1, max='max')
        self.spinbox_streamer_buffer.valueChanged.connect(self.set_streamer_buffer)

        self.spinbox_processor_buffer = SpinBox(
            name='samples', layout_list=settings_items,
            type=int, value=self.settings['processor_buffer'], step=1, max='max')
        self.spinbox_processor_buffer.valueChanged.connect(self.set_processor_buffer)


        self.spinbox_shaker_amplitude = SpinBox(
            name='Shaker Amplitude', layout_list=settings_items,
            type=float, value=self.settings['shaker_amplitude'], step=.01, suffix='ps')
        self.spinbox_shaker_amplitude.valueChanged.connect(self.set_shaker_amplitude)

        for item in settings_items:
            settings_box_layout.addWidget(labeled_qitem(*item))
        self.label_processor_fps = QLabel('FPS: 0')
        settings_box_layout.addWidget(self.label_processor_fps)
        self.label_streamer_fps = QLabel('FPS: 0')
        settings_box_layout.addWidget(self.label_streamer_fps)
        self.radio_dark_control = QRadioButton('Dark Control')
        self.radio_dark_control.setChecked(True)
        settings_box_layout.addWidget(self.radio_dark_control)
        self.toggle_darkcontrol_mode()

        self.apply_settings_button = QPushButton('Apply')
        settings_box_layout.addWidget(self.apply_settings_button)
        self.apply_settings_button.clicked.connect(self.apply_settings)


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

    def make_data_manager(self, processor_buffer=30000, streamer_buffer=100000):
        data_manager = DataManager(processor_buffer=processor_buffer, streamer_buffer=streamer_buffer)
        data_manager.newProcessedData.connect(self.on_processed_data)
        data_manager.newStreamerData.connect(self.on_streamer_data)

        data_manager_thread = Thread()
        data_manager.moveToThread(data_manager_thread)
        data_manager_thread.start()
        # data_manager_thread.started.connect(data_manager.__init__())

        return data_manager, data_manager_thread

    def toggle_simulation_mode(self):
        self.data_manager.toggle_simulation(self.radio_simulate.isChecked())

    def toggle_darkcontrol_mode(self):
        self.data_manager.toggle_darkcontrol(self.radio_dark_control.isChecked())

    def on_main_clock(self):
        pass
        # x = np.linspace(0, 99, 100)
        # y = np.random.rand(100)
        # # self.visual_widget.add_main_plot_line('test',(255,255,255))
        # self.visual_widget.plot_main('test', x, y)

    def on_processed_data(self,data):
        try:
            t0=self.processor_tick
            self.processor_tick = time.time()

            if len(self.fps_l) >=100:
                self.fps_l.pop(0)
                print('\n\n\n\n{}\n\n\n\n'.format(self.fps_l))

            self.fps_l.append(1./(self.processor_tick-t0))
            fps = np.mean(self.fps_l)
            self.label_processor_fps.setText('FPS: {:.2f}'.format(fps))
        except:
            self.processor_tick = time.time()

        self.visual_widget.plot_main('processed signal', data[0], data[1])

        print('recieved processed data')

    def on_streamer_data(self,data):
        n_samples = data.shape[1]
        x=np.linspace(0,n_samples-1,n_samples)
        self.visual_widget.plot_secondary('stage pos', x=x, y=data[0])
        self.visual_widget.plot_secondary('raw signal',x=x, y=data[1])


    def start_acquisition(self):
        self.data_manager.create_streamer()
        self.data_manager.start_streamer()

    def stop_acquisition(self):
        self.data_manager.stop_streamer()

    def reset_data(self):
        self.data_manager.reset_data()

    def set_n_samples(self):
        pass

    def set_shaker_amplitude(self,val):
        pass
    def set_processor_buffer(self,val):
        pass
    def set_streamer_buffer(self,val):
        pass
    def save_data(self):
        pass

    def apply_setings(self):
        try:
            streamer_buffer = self.spinbox_streamer_buffer.value()
            self.data_manager.streamer_buffer_size = streamer_buffer
            processor_buffer = self.spinbox_processor_buffer.value()
            self.data_manager.processor_buffer_size = processor_buffer

            shaker_amplitude = self.spinbox_streamer_buffer.value()
            self.data_manager.shaker_amplitude = shaker_amplitude

            dark_control = self.radio_dark_control.isChecked()
            self.data_manager.toggle_darkcontrol(dark_control)

            self.settings = {'streamer_buffer': streamer_buffer,
                             'processor_buffer':processor_buffer,
                             'shaker_amplitude': shaker_amplitude,
                             'dark_control':dark_control
                             }
        except Exception as e:
            print(e)

    def closeEvent(self,event):
        super(FastScanMainWindow, self).closeEvent(event)

        print('attempting to close')

        self.data_manager.close()#todo: fix closing issue
        self.data_manager_thread.exit()


class FastScanPlotWidget(QWidget):

    def __init__(self):
        super(FastScanPlotWidget, self).__init__()

        self.main_plot_lines = {}
        self.secondary_plot_lines = {}
        self.clock = QTimer()
        self.clock.setInterval(1. / 60)
        self.clock.timeout.connect(self.on_clock)
        self.clock.start()

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.main_plot_widget = pg.PlotWidget(name='raw_data_plot')
        self.setup_plot_widget(self.main_plot_widget, title='Signal')
        # self.main_legend = pg.LegendItem() #TODO: find a way to add legends
        # self.main_legend.setParentItem(self.main_plot_widget.getPlotItem())

        self.secondary_plot_widget = pg.PlotWidget(name='raw_data_plot')
        self.setup_plot_widget(self.secondary_plot_widget, title='raw data stream')

        self.raw_data_plot = self.secondary_plot_widget.plot()
        self.raw_data_plot.setPen(pg.mkPen(255, 255, 255))

        vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        vsplitter.addWidget(self.main_plot_widget)
        vsplitter.addWidget(self.secondary_plot_widget)

        layout.addWidget(vsplitter)

    def setup_plot_widget(self, plot_widget, title='Plot'):
        plot_widget.showAxis('top', True)
        plot_widget.showAxis('right', True)
        plot_widget.showGrid(True, True, .2)
        plot_widget.setLabel('left', 'Value', units='V')
        plot_widget.setLabel('bottom', 'Time', units='s')
        # plot_widget.setLabel('top', title)

    def resizeEvent(self, event):
        h = self.frameGeometry().height()
        w = self.frameGeometry().width()

        self.main_plot_widget.setMinimumHeight(int(h * .7))
        self.main_plot_widget.setMinimumWidth(500)

    def add_main_plot_line(self, name, pen):
        if name not in self.main_plot_lines.keys():
            plot_line = self.main_plot_widget.plot()
            if isinstance(pen, tuple):
                pen = pg.mkPen(pen)
            plot_line.setPen(pen)
            self.main_plot_lines[name] = {'plot': plot_line}
        else:
            self.main_plot_lines[name]['plot'].setPen(pen)

    def add_secondary_plot_line(self, name, pen):
        if name not in self.secondary_plot_lines.keys():
            plot_line = self.secondary_plot_widget.plot()
            if isinstance(pen, tuple):
                pen = pg.mkPen(pen)
            plot_line.setPen(pen)
            self.secondary_plot_lines[name] = {'plot': plot_line}
        else:
            self.secondary_plot_lines[name]['plot'].setPen(pen)

    def plot_main(self, name, x=None, y=None):
        if name not in self.main_plot_lines.keys():
            print('adding curve {}'.format(name))
            self.add_main_plot_line(name, (255, 255, 255))
        self.main_plot_lines[name]['x'] = x
        self.main_plot_lines[name]['y'] = y

    def plot_secondary(self, name, x=None, y=None):
        if name not in self.secondary_plot_lines.keys():
            print('adding curve {}'.format(name))
            self.add_secondary_plot_line(name, (255, 255, 255))
        self.secondary_plot_lines[name]['x'] = x
        self.secondary_plot_lines[name]['y'] = y

    def on_clock(self):
        try:
            for key, val in self.main_plot_lines.items():
                val['plot'].setData(val['x'], val['y'])
            for key, val in self.secondary_plot_lines.items():
                val['plot'].setData(val['x'], val['y'])
        except:
            print('failed plotting')

def main():
    pass


if __name__ == '__main__':
    main()
