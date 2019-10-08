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
import os
import time

import numpy as np
import pyqtgraph as pg
import qdarkstyle
import xarray as xr
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QDoubleSpinBox, \
    QLineEdit, QComboBox, QSizePolicy, \
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox, QPushButton, QGridLayout, QSpinBox, QLabel
from pyqtgraph.Qt import QtCore as pQtCore, QtGui as pQtGui
from scipy.signal import butter, filtfilt

from .fastscan import FastScanThreadManager
from .misc import parse_category, parse_setting, write_setting, NoDataException, labeledQwidget


class FastScanMainWindow(QMainWindow):

    def __init__(self):
        super(FastScanMainWindow, self).__init__()
        self.logger = logging.getLogger('{}.FastScanMainWindow'.format(__name__))
        self.logger.info('Created MainWindow')

        self.setWindowTitle('Fast Scan')
        width = parse_setting('gui', 'width')
        height = parse_setting('gui', 'height')
        hpos = parse_setting('gui', 'hpos')
        vpos = parse_setting('gui', 'vpos')

        self.setGeometry(hpos, vpos, width, height)
        # self.showMaximized()

        self.setWindowIcon(QtGui.QIcon('logo256_w.png'))

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('ready')
        #######################################################
        # set the cool dark theme and other plotting settings #
        #######################################################
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        # self.setStyleSheet("QLabel {font: 12pt}")

        # self.setStyleSheet('QWidget {font: "Arial"}')

        pg.setConfigOption('background', (25, 35, 45))
        pg.setConfigOption('foreground', (255, 255, 255))  # 'w')
        pg.setConfigOptions(antialias=True)

        ##########################
        #   define attributes    #
        ##########################

        self.settings = parse_category('fastscan')  # import all

        self.data_manager = FastScanThreadManager()
        self.data_manager.newProcessedData.connect(self.on_processed_data)
        self.data_manager.newStreamerData.connect(self.on_streamer_data)
        self.data_manager.newFitResult.connect(self.on_fit_result)
        self.data_manager.newAverage.connect(self.on_avg_data)
        # self.data_manager.newData.connect(self.on_new_data)
        self.data_manager.completedIterativeMeasurement.connect(self.on_completed_iterative_measurement)
        self.data_manager.error.connect(self.on_thread_error)

        self.data_manager_thread = QtCore.QThread()
        self.data_manager.moveToThread(self.data_manager_thread)
        self.data_manager_thread.start()

        self._popup_enabled = True

        self.fps_l = []
        self.streamer_qsize = 0

        self.main_clock = QtCore.QTimer()
        self.main_clock.setInterval(50)
        self.main_clock.start()
        self.main_clock.timeout.connect(self.on_main_clock)

        self.setupUi()
        self.show()

    def setupUi(self):

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)

        control_widget = self.make_controlwidget()
        sh = control_widget.sizeHint()
        # control_widget.resize(1,sh.height())
        control_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

        self.visual_widget = FastScanPlotWidget()
        sh = self.visual_widget.sizeHint()
        self.visual_widget.resize(800, 600)
        self.visual_widget.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(control_widget)
        main_splitter.addWidget(self.visual_widget)

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
        self.start_button.clicked.connect(self.start_acquisition)
        self.start_button.setEnabled(True)

        self.stop_button = QPushButton('stop')
        acquisition_box_layout.addWidget(self.stop_button, 0, 1, 1, 1)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)

        self.reset_button = QPushButton('reset')
        acquisition_box_layout.addWidget(self.reset_button, 1, 1, 1, 1)
        self.reset_button.clicked.connect(self.reset_data)
        self.reset_button.setEnabled(True)

        self.n_averages_spinbox = QSpinBox()
        self.n_averages_spinbox.setMinimum(1)
        self.n_averages_spinbox.setMaximum(999999)

        self.n_averages_spinbox.setValue(parse_setting('fastscan', 'n_averages'))
        self.n_averages_spinbox.valueChanged[int].connect(lambda x: write_setting(x, 'fastscan', 'n_averages'))

        acquisition_box_layout.addWidget(QLabel('Averages: '), 2, 0, 1, 1)
        acquisition_box_layout.addWidget(self.n_averages_spinbox, 2, 1, 1, 1)

        # ----------------------------------------------------------------------
        # Save Box
        # ----------------------------------------------------------------------

        save_box = QGroupBox('Save')
        savebox_layout = QGridLayout()
        layout.addWidget(save_box)

        save_box.setLayout(savebox_layout)
        h5_dir = parse_setting('paths', 'h5_data')
        f_name = parse_setting('paths', 'filename')

        self.save_name_ledit = QLineEdit(f_name)
        savebox_layout.addWidget(QLabel('File Name:'), 0, 0)
        savebox_layout.addWidget(self.save_name_ledit, 0, 1)
        self.save_name_ledit.editingFinished.connect(self.update_savename)

        self.save_dir_ledit = QLineEdit(h5_dir)
        savebox_layout.addWidget(QLabel('Directory:'), 1, 0)
        savebox_layout.addWidget(self.save_dir_ledit, 1, 1)
        self.save_dir_ledit.editingFinished.connect(self.update_savedir)

        self.save_all_cb = QCheckBox('Save avgs')
        self.save_all_cb.setToolTip(
            'If Checked, saves all projected average curves, \nelse it only saves the accumulate average trace.')
        savebox_layout.addWidget(self.save_all_cb, 2, 0)
        self.save_all_cb.setChecked(True)

        self.save_data_button = QPushButton('Save')
        savebox_layout.addWidget(self.save_data_button, 2, 1, 2, 2)
        self.save_data_button.clicked.connect(self.save_data)

        # self.autosave_checkbox = QCheckBox('autosave')
        # savebox_layout.addWidget(self.autosave_checkbox,3,0)
        # self.autosave_timeout = QDoubleSpinBox()
        # self.autosave_timeout.setValue(1)
        # savebox_layout.addWidget(QLabel('timeout (s)'),3,1)
        # savebox_layout.addWidget(self.autosave_timeout,3,2)

        self.datasize_label = QLabel('data Size')
        savebox_layout.addWidget(self.datasize_label, 4, 0, 3, 2)
        # self.fps_label = QLabel('data Size')
        # savebox_layout.addWidget(self.fps_label, 6, 0, 3, 2)

        # ----------------------------------------------------------------------
        # Settings Box
        # ----------------------------------------------------------------------
        settings_box = QGroupBox('settings')
        layout.addWidget(settings_box)
        settings_box_layout = QGridLayout()
        settings_box.setLayout(settings_box_layout)

        settings_items = []

        self.spinbox_n_samples = QSpinBox()
        self.spinbox_n_samples.setMaximum(2147483647)
        self.spinbox_n_samples.setMinimum(100)
        self.spinbox_n_samples.setValue(parse_setting('fastscan', 'n_samples'))
        self.spinbox_n_samples.setSingleStep(100)
        self.spinbox_n_samples.valueChanged.connect(self.set_n_samples)
        settings_box_layout.addWidget(QLabel('n° of samples'), 0, 0)
        settings_box_layout.addWidget(self.spinbox_n_samples, 0, 1)

        self.label_processor_fps = QLabel('FPS: 0')
        # settings_box_layout.addWidget(self.label_processor_fps)

        self.shaker_gain_combobox = QComboBox()
        self.shaker_gain_combobox.addItem('1')
        self.shaker_gain_combobox.addItem('10')
        self.shaker_gain_combobox.addItem('100')
        idx = self.shaker_gain_combobox.findText(str(parse_setting('fastscan', 'shaker_gain')))
        if idx == -1:
            self.shaker_gain_combobox.setCurrentIndex(1)
        else:
            self.shaker_gain_combobox.setCurrentIndex(idx)

        def set_shaker_gain(val):
            self.data_manager.shaker_gain = val

        self.shaker_gain_combobox.activated[str].connect(set_shaker_gain)
        settings_box_layout.addWidget(QLabel('Shaker Gain'), 1, 0)
        settings_box_layout.addWidget(self.shaker_gain_combobox, 1, 1)

        # self.filter_frequency_spinbox.setMaximum(1.)
        # self.filter_frequency_spinbox.setMinimum(0.0)

        # self.apply_settings_button.clicked.connect(self.apply_settings)

        # self.apply_settings_button = QPushButton('Apply')
        # settings_box_layout.addWidget(self.apply_settings_button)

        # ----------------------------------------------------------------------
        # Filter Box
        # ----------------------------------------------------------------------
        # filter_box = QGroupBox('Filter')
        # layout.addWidget(filter_box)
        # filter_box_layout = QGridLayout()
        # filter_box.setLayout(filter_box_layout)
        #
        # self.butter_filter_checkbox = QCheckBox('Butter')
        # filter_box_layout.addWidget(self.butter_filter_checkbox, 0, 0)
        # self.filter_order_spinbox = QSpinBox()
        # self.filter_order_spinbox.setValue(2)
        #
        # filter_box_layout.addWidget(QLabel('Order:'), 0, 1)
        # filter_box_layout.addWidget(self.filter_order_spinbox, 0, 2)
        #
        # self.filter_frequency_spinbox = QDoubleSpinBox()
        # self.filter_frequency_spinbox.setValue(.3)
        # self.filter_frequency_spinbox.setMaximum(1.)
        # self.filter_frequency_spinbox.setMinimum(0.0)
        # self.filter_frequency_spinbox.setSingleStep(0.1)
        #
        # filter_box_layout.addWidget(QLabel('Cut (0.-1.):'), 0, 3)
        # filter_box_layout.addWidget(self.filter_frequency_spinbox, 0, 4)
        # self.butter_filter_checkbox.setChecked(False)

        # ----------------------------------------------------------------------
        # Autocorrelation Box
        # ----------------------------------------------------------------------

        autocorrelation_box = QGroupBox('Autocorrelation')
        autocorrelation_box_layout = QGridLayout()
        autocorrelation_box.setLayout(autocorrelation_box_layout)

        self.calculate_autocorrelation_box = QCheckBox('Fit')
        autocorrelation_box_layout.addWidget(self.calculate_autocorrelation_box)
        self.calculate_autocorrelation_box.setChecked(False)
        self.calculate_autocorrelation_box.clicked.connect(self.toggle_calculate_autocorrelation)

        self.fit_n_wings = QSpinBox()
        self.fit_n_wings.setValue(2)
        self.fit_n_wings.valueChanged[int].connect(lambda x: write_setting(x, 'autocorrelation guess', 'n_wings'))

        font = QFont()
        font.setBold(True)
        font.setPointSize(16)
        # report = '{:^8}|{:^8}|{:^8}|{:^8}\n{:^8.3f}|{:^8.3f}|{:^8.3f}|{:^8.3f}'.format(
        #     'Amp', 'Xc', 'FWHM', 'off', .0, .0, .0, .0)
        # self.autocorrelation_report_label = QLabel(report)

        self.pulse_duration_label = QLabel('0 fs')
        self.pulse_duration_label.setFont(font)

        autocorrelation_box_layout.addWidget(self.calculate_autocorrelation_box, 0, 0, 1, 1)
        # autocorrelation_box_layout.addWidget(self.autocorrelation_report_label, 0, 1, 1, 2)
        autocorrelation_box_layout.addWidget(labeledQwidget('n of wings', self.fit_n_wings), 0, 1)
        autocorrelation_box_layout.addWidget(QLabel('Pulse duration:'), 2, 0, 1, 1)
        autocorrelation_box_layout.addWidget(self.pulse_duration_label, 2, 1, 1, 2)

        layout.addWidget(autocorrelation_box)

        # ----------------------------------------------------------------------
        # Stage Control Box
        # ----------------------------------------------------------------------

        # self.delay_stage_widget = DelayStageWidget(self.data_manager.delay_stage)
        # # layout.addWidget(self.delay_stage_widget)
        #
        #
        # shaker_calib_gbox = QGroupBox('Shaker Calibration')
        # shaker_calib_layout = QGridLayout()
        # shaker_calib_gbox.setLayout(shaker_calib_layout)
        # self.shaker_calib_btn = QPushButton('Shaker Calibration')
        # shaker_calib_layout.addWidget(self.shaker_calib_btn,0,0,2,2)
        # self.shaker_calib_btn.clicked.connect(self.on_shaker_calib)
        # self.shaker_calib_iterations = QSpinBox()
        # self.shaker_calib_iterations.setValue(50)
        # self.shaker_calib_iterations.setMinimum(4)
        # self.shaker_calib_iterations.setMaximum(100000)
        # self.shaker_calib_integration = QSpinBox()
        # self.shaker_calib_integration.setValue(5)
        # self.shaker_calib_integration.setMinimum(1)
        # self.shaker_calib_integration.setMaximum(100000)
        #
        # shaker_calib_layout.addWidget(QLabel('iterations:'),0,2,1,1)
        # shaker_calib_layout.addWidget(QLabel('integrations:'),1,2,1,1)
        # shaker_calib_layout.addWidget(self.shaker_calib_iterations,0,3,1,1)
        # shaker_calib_layout.addWidget(self.shaker_calib_integration,1,3,1,1)

        # ----------------------------------------------------------------------
        # Iterative Measurement Box
        # ----------------------------------------------------------------------

        iterative_measurement_box = QGroupBox('Iterative Measurement')
        iterative_measurement_box_layout = QGridLayout()
        iterative_measurement_box.setLayout(iterative_measurement_box_layout)
        layout.addWidget(iterative_measurement_box)

        self.im_save_name = QLineEdit('measurement session name')
        iterative_measurement_box_layout.addWidget(QLabel('Name:'), 0, 0)
        iterative_measurement_box_layout.addWidget(self.im_save_name, 0, 1)
        self.im_save_dir = QLineEdit('D:\\')
        iterative_measurement_box_layout.addWidget(QLabel('dir :'), 1, 0)
        iterative_measurement_box_layout.addWidget(self.im_save_dir, 1, 1)
        self.im_temperatures = QLineEdit('5,10,15,20')
        iterative_measurement_box_layout.addWidget(QLabel('temperatures :'), 2, 0)
        iterative_measurement_box_layout.addWidget(self.im_temperatures, 2, 1)

        self.start_iterative_measurement_button = QPushButton('Start')
        iterative_measurement_box_layout.addWidget(self.start_iterative_measurement_button, 3, 0, 1, 2)
        self.start_iterative_measurement_button.clicked.connect(self.start_iterative_measurement)

        # layout.addWidget(shaker_calib_gbox)
        layout.addStretch()
        return widget

    def keyPressEvent(self, e):
        """ Manage Hotkeys"""
        if e.key() == QtCore.Qt.Key_F11:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()

    @QtCore.pyqtSlot()
    def update_savename(self):
        name = self.save_name_ledit.text()
        write_setting(name, 'paths', 'filename')

    @QtCore.pyqtSlot()
    def update_savedir(self):
        name = self.save_dir_ledit.text()
        write_setting(name, 'paths', 'h5_data')

    @QtCore.pyqtSlot()
    def start_iterative_measurement(self):
        self._popup_enabled = False
        temperatures = [float(x) for x in self.im_temperatures.text().split(',')]
        savename = os.path.join(self.im_save_dir.text(), self.im_save_name.text())
        self.data_manager.start_iterative_measurement(temperatures, savename)

    @QtCore.pyqtSlot()
    def on_main_clock(self):
        # self.main_clock.setInterval(self.autosave_timeout.value())
        try:
            streamer_shape = self.data_manager.streamer_average.shape
            projected_shape = self.data_manager.all_curves.shape
        except AttributeError:
            streamer_shape = projected_shape = (0, 0)
        try:
            if len(self.fps_l) > 10:
                self.fps_l.pop(0)
            fps = np.mean(self.fps_l)
        except:
            fps = 0

        string = 'Data Size :\n streamer: {} - {:10.3f} Kb\n projected: {} - {:10.3f} Kb\n Queues: Stream: {} Projected: {}\n Cycles per Second [Hz]: {:10.3f}       '.format(
            streamer_shape, np.prod(streamer_shape) / (1024),
            projected_shape, np.prod(projected_shape) / (1024),
            self.data_manager.stream_qsize,
            self.data_manager.processed_qsize,
            fps,

        )
        self.datasize_label.setText(string)

    @QtCore.pyqtSlot()
    def toggle_calculate_autocorrelation(self):
        self.data_manager._calculate_autocorrelation = self.calculate_autocorrelation_box.isChecked()

    @QtCore.pyqtSlot()
    def on_shaker_calib(self):
        self.data_manager.calibrate_shaker(self.shaker_calib_iterations.value(), self.shaker_calib_integration.value())

    @QtCore.pyqtSlot(xr.DataArray)
    def on_processed_data(self, da):
        try:
            t0 = self.processor_tick
            self.processor_tick = time.time()
            if len(self.fps_l) >= 100:
                self.fps_l.pop(0)
            self.fps_l.append(1. / (self.processor_tick - t0))
            # fps = np.mean(self.fps_l)
            # self.fps_label.setText('Cycles (Hz): {:.2f}'.format(fps))
        except:
            self.processor_tick = time.time()
        # f_data = self.apply_filter(da)
        self.visual_widget.plot_last_curve(da)
        self.logger.debug('recieved processed data as {}'.format(type(da)))

    @QtCore.pyqtSlot(dict)
    def on_fit_result(self, fitDict):
        self.pulse_duration_label.setText('{:.3f} ps'.format(fitDict['popt'][2] * .65))
        # report = '{:^8}|{:^8}|{:^8}|{:^8}\n{:^8.3f}|{:^8.3f}|{:^8.3f}|{:^8.3f}'.format(
        #     'Amp', 'Xc', 'FWHM', 'off', *fitDict['popt'])
        # self.autocorrelation_report_label.setText(report)

        self.visual_widget.plot_fit_curve(fitDict['curve'])

    @QtCore.pyqtSlot(xr.DataArray)
    def on_avg_data(self, da):
        assert isinstance(da, xr.DataArray) and da.ndim == 1, 'must be 1D, found {}'.fromat(da.ndim)
        # f_data = self.apply_filter(da)
        self.visual_widget.plot_avg_curve(da)

    # @QtCore.pyqtSlot(xr.DataArray)
    # def on_new_data(self, da):
    #     assert isinstance(da, xr.DataArray)
    #     self.logger.debug('calculating avg for plotting')
    #     if 'avg' in da.dims:
    #         def f(da):
    #             return da.mean('avg').dropna('time')
    #
    #         runnable = Runnable(f, da)
    #         self.pool.start(runnable)
    #         runnable.signals.result.connect(self.on_avg_data)
    #

    @QtCore.pyqtSlot(np.ndarray)
    def on_streamer_data(self, data):
        self.visual_widget.plot_stream_curve(data)

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # self.data_manager.create_streamer()
        self.status_bar.showMessage('Acquisition started')
        self.data_manager.start_streamer()

    @QtCore.pyqtSlot()
    def stop_acquisition(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_bar.showMessage('Acquisition Stopped')
        self.data_manager.stop_streamer()

    @QtCore.pyqtSlot()
    def reset_data(self):
        self.status_bar.showMessage('Data reset')
        self.fps_l = []
        self.data_manager.reset_data()

    @QtCore.pyqtSlot()
    def set_n_samples(self, var):
        self.data_manager.n_samples = var

    @QtCore.pyqtSlot(int)
    def set_n_averages(self, val):
        self.data_manager.n_averages = val

    @QtCore.pyqtSlot()
    def save_data(self):
        self.logger.debug('Saving data...')

        dir = self.save_dir_ledit.text()
        filename = self.save_name_ledit.text()
        filepath = os.path.join(dir, filename)
        filepath += '.h5'

        try:
            self.data_manager.save_data(filepath, all_data=self.save_all_cb.isChecked())
            self.status_bar.showMessage('Successfully saved data as {}'.format(filepath))
        except NotADirectoryError:
            self.logger.debug('Directory "{}" not found'.format(dir))
            reply = QtWidgets.QMessageBox.question(self,
                                                   'No directory {}\nCreate it?'.format(dir),
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.Yes)
            if reply == QtWidgets.QMessageBox.Yes:
                self.logger.debug('Directory {} created'.format(dir))
                os.mkdir(dir)
                self.save_data()  # restart saving procedure
            else:
                self.logger.debug('Cancelled saving'.format(dir))

        except FileExistsError:
            text, okPressed = QtWidgets.QInputDialog.getText(self,
                                                             'File Exists Error',
                                                             'File {} already exists! need to set a new name.'.format(
                                                                 filepath),
                                                             QLineEdit.Normal,
                                                             filename)
            if okPressed and text != '':
                self.save_name_ledit.setText(text)
                self.save_data()  # restart saving procedure
            else:
                self.logger.debug('Saving aborted'.format(dir))

        except NoDataException:
            self.logger.debug('Attempting to save nothing...'.format(filepath))
            errorDialog = QtWidgets.QMessageBox()
            errorDialog.setText('No data to save...')
            errorDialog.setIcon(QtWidgets.QMessageBox.Warning)
            errorDialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
            errorDialog.exec_()

        except Exception as e:
            raise e

    @QtCore.pyqtSlot()
    def on_completed_iterative_measurement(self):
        self._popup_enabled = True
        infoBox = QtWidgets.QMessageBox()
        infoBox.setText('Iterative measurement completed!')
        infoBox.setIcon(QtWidgets.QMessageBox.Info)
        infoBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        infoBox.exec_()

    @QtCore.pyqtSlot(Exception)
    def on_thread_error(self, e):
        self.logger.critical('Thread error: {}'.format(e))
        raise e

    def closeEvent(self, event):
        super(FastScanMainWindow, self).closeEvent(event)
        self.logger.info('Closing window: terminating all threads.')
        self.data_manager.close()
        self.data_manager_thread.exit()

    def resizeEvent(self, event):
        write_setting(event.size().width(), 'gui', 'width')
        write_setting(event.size().height(), 'gui', 'height')

    def moveEvent(self, event):
        write_setting(event.pos().x(), 'gui', 'hpos')
        write_setting(event.pos().y(), 'gui', 'vpos')


class FastScanPlotWidget(QWidget):
    """ Class to manage the plotting section of the GUI."""

    def __init__(self):
        super(FastScanPlotWidget, self).__init__()
        self.logger = logging.getLogger('-.{}.PlotWidget'.format(__name__))
        self.logger.info('Created PlotWidget')
        self.make_layout()

        self.curve_std, self.avg_std, self.avg_max = 1, 1, 1
        self.use_r0 = parse_setting('fastscan', 'use_r0')
        self._scale_follow = False
        self._main_plot_ranges = {'Xr_min': [],
                                  'Xr_max': [],
                                  'Yr_min': [],
                                  'Yr_max': []}

        self.clock = QTimer()
        self.clock.setInterval(1000. / 30)
        self.clock.timeout.connect(self._on_clock)
        self.clock.start()

        self.curves = {}

    def make_layout(self):
        """ take care of the gui layout."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.main_plot_frame = QtWidgets.QFrame()
        layout_mpf = QVBoxLayout()
        self.main_plot_frame.setLayout(layout_mpf)

        btn_area = QWidget()
        l = QHBoxLayout()
        btn_area.setLayout(l)
        self.btn_autoscale = QPushButton('Autoscale')
        self.btn_autoscale.clicked.connect(self.set_autoscale)
        self.btn_showall = QPushButton('Fixed')
        self.btn_showall.clicked.connect(self.set_showall)
        self.btn_follow = QPushButton('Follow')
        self.btn_follow.clicked.connect(self.set_follow)
        l.addStretch()
        l.addWidget(self.btn_autoscale)
        l.addWidget(self.btn_showall)
        l.addWidget(self.btn_follow)
        l.addStretch()

        self.main_plot_widget = pg.PlotWidget(name='main_plot')
        self.main_plot_widget.setBackground((25, 25, 25))
        self.main_plot_widget.setAntialiasing(True)

        layout_mpf.addWidget(self.main_plot_widget)
        layout_mpf.addWidget(btn_area)

        self.main_plot = self.main_plot_widget.getPlotItem()
        self.main_plot.showAxis('top', True)
        self.main_plot.showAxis('right', True)
        self.main_plot.showGrid(True, True, .2)
        self.main_plot.setLabel('left', '<font>&Delta;R</font>', units='V')
        self.main_plot.setLabel('left', '<font>&Delta;R / R</font>', units='%')
        self.main_plot.setLabel('bottom', 'Time', units='s')
        self.set_autoscale()

        self.small_plot_widget = pg.PlotWidget(name='stream_plot')
        self.small_plot_widget.setMaximumWidth(400)
        self.small_plot_widget.setMinimumWidth(200)

        self.small_plot = self.small_plot_widget.getPlotItem()
        self.small_plot.showAxis('top', True)
        self.small_plot.showAxis('right', True)
        self.small_plot.showGrid(True, True, .2)
        self.small_plot.setLabel('left', 'Value', units='V')
        self.small_plot.setLabel('bottom', 'Time', units='samples')

        curve_list = QGroupBox('Curves')
        curve_list_layout = QVBoxLayout()
        curve_list.setLayout(curve_list_layout)

        self.cb_last_curve = QCheckBox('last curve')
        curve_list_layout.addWidget(self.cb_last_curve)
        self.cb_last_curve.setChecked(False)
        self.cb_avg_curve = QCheckBox('average curve')
        curve_list_layout.addWidget(self.cb_avg_curve)
        self.cb_avg_curve.setChecked(True)
        self.cb_fit_curve = QCheckBox('fit curve')
        curve_list_layout.addWidget(self.cb_fit_curve)
        self.cb_fit_curve.setChecked(True)

        self.clean_plot_btn = QPushButton('Clean plot')
        self.clean_plot_btn.clicked.connect(self.clean_plot)
        curve_list_layout.addWidget(self.clean_plot_btn)

        controls = QGroupBox('Plot Settings')
        controls_layout = QVBoxLayout()
        controls.setLayout(controls_layout)

        self.cb_remove_baseline = QCheckBox('Remove Baseline')
        controls_layout.addWidget(self.cb_remove_baseline)
        self.cb_remove_baseline.setChecked(True)

        self.avg_side_cutoff = QSpinBox()
        self.avg_side_cutoff.setValue(0)
        self.avg_side_cutoff.setMinimum(0)
        controls_layout.addWidget(labeledQwidget('Side Cutoff:', self.avg_side_cutoff))
        self.noise_label = QLabel('Noise Floor: 0')
        controls_layout.addWidget(self.noise_label)

        filter_box = QWidget()
        controls_layout.addWidget(filter_box)
        filter_box_layout = QHBoxLayout()
        filter_box.setLayout(filter_box_layout)
        self.butter_filter_checkbox = QCheckBox('Butter')
        self.filter_order_spinbox = QSpinBox()
        self.filter_order_spinbox.setValue(2)
        self.filter_frequency_spinbox = QDoubleSpinBox()
        self.filter_frequency_spinbox.setValue(.3)
        self.filter_frequency_spinbox.setMaximum(1.)
        self.filter_frequency_spinbox.setMinimum(0.0)
        self.filter_frequency_spinbox.setSingleStep(0.1)
        self.butter_filter_checkbox.setChecked(False)

        filter_box_layout.addWidget(self.butter_filter_checkbox)
        filter_box_layout.addWidget(labeledQwidget('Order', self.filter_order_spinbox, align='v'))
        filter_box_layout.addWidget(labeledQwidget('Cut (0.-1.):', self.filter_frequency_spinbox, align='v'))
        filter_box_layout.addStretch()

        curve_list_layout.addStretch()
        controls_layout.addStretch()

        vsplitter = pQtGui.QSplitter(pQtCore.Qt.Vertical)
        hsplitter = pQtGui.QSplitter(pQtCore.Qt.Horizontal)
        vsplitter.addWidget(self.main_plot_frame)
        vsplitter.addWidget(hsplitter)
        # vsplitter.setSizes([100,1])
        vsplitter.setStretchFactor(0, 1.7)
        hsplitter.addWidget(curve_list)
        hsplitter.addWidget(controls)
        hsplitter.addWidget(self.small_plot_widget)


        # vsplitter.setStretchFactor(0,3)
        layout.addWidget(vsplitter)

    def resizeEvent(self, event):
        """ Impose gemometry to optimize plot window size."""
        h = self.frameGeometry().height()
        w = self.frameGeometry().width()
        self.main_plot_frame.setMinimumHeight(int(h * .7))
        self.main_plot_frame.setMinimumWidth(500)

    @QtCore.pyqtSlot()
    def _on_clock(self):
        """ """
        if self._scale_follow:
            self.lazy_autoscale(100)

    def add_curve(self, name, *args, **kwargs):
        """ Append a plotItem to the list of curves for the main plot."""
        self.curves[name] = self.main_plot.plot(name=name)
        self.curves[name].setPen(*args, **kwargs)

    def draw_curve(self, name, da):
        """ Set data to the indicated curve

        :param name: str
            name of the curve
        :param da: xr.DataArray
            data to set
        """
        if name in self.curves:
            if self.use_r0:
                self.main_plot.setLabel('left', '<font>&Delta;R / R</font>', units='')
                self.curves[name].setData(da.time * 10 ** -12, da)  # *100) # uncomment to represent in %
            else:
                self.main_plot.setLabel('left', '<font>&Delta;R</font>', units='V')
                self.curves[name].setData(da.time * 10 ** -12, da)

    @QtCore.pyqtSlot()
    def plot_last_curve(self, da):
        """ Plots data from the last available projected dataset"""
        if self.cb_last_curve.isChecked():
            if 'last' not in self.curves:
                self.add_curve('last', color=(200, 200, 200), alpha=.1)
            da_ = self.post_process(da)
            self.draw_curve('last', da_)
        else:
            if 'last' in self.curves:
                self.main_plot.removeItem(self.curves.pop('last'))

    @QtCore.pyqtSlot()
    def plot_avg_curve(self, da):
        """ Plots data from the last available averaged dataset"""
        if self.cb_avg_curve.isChecked():
            if 'avg' not in self.curves:
                self.add_curve('avg', color=(255, 100, 100), width=2)

            self._calculate_noise_floor(da)
            da_ = self.post_process(da, cutEdges=True)
            self.draw_curve('avg', da_)  # plot the curve
        else:
            if 'avg' in self.curves:
                self.main_plot.removeItem(self.curves.pop('avg'))

    @QtCore.pyqtSlot()
    def clean_plot(self):
        if not self.cb_last_curve.isChecked() and 'last' in self.curves:
            self.main_plot.removeItem(self.curves.pop('last'))
        if not self.cb_avg_curve.isChecked() and 'avg' in self.curves:
            self.main_plot.removeItem(self.curves.pop('avg'))
        if not self.cb_fit_curve.isChecked() and 'fit' in self.curves:
            self.main_plot.removeItem(self.curves.pop('fit'))

    @QtCore.pyqtSlot()
    def plot_fit_curve(self, da):
        """ Plots data from the last available Fit curve dataset"""

        if self.cb_fit_curve.isChecked():
            if 'fit' not in self.curves:
                self.add_curve('fit', color=(100, 255, 100))
            da_ = self.post_process(da, filter=False)
            self.draw_curve('fit', da_)
        else:
            if 'fit' in self.curves:
                self.main_plot.removeItem(self.curves.pop('fit'))

    @QtCore.pyqtSlot()
    def plot_stream_curve(self, data):
        """ plot the significant channels from the streamer data"""

        # check if we have r0, and change main plot labels accordingly
        if parse_setting('fastscan', 'use_r0') and data.shape[0] == 4:
            self.use_r0 = True

        if not hasattr(self, 'stream_curve'):
            self.stream_curve = self.small_plot.plot()
            self.stream_curve.setPen((pg.mkPen(255, 100, 100)))
            self.stream_signal_dc0 = self.small_plot.plot()
            self.stream_signal_dc0.setPen((pg.mkPen(100, 255, 100)))
            self.stream_signal_dc1 = self.small_plot.plot()
            self.stream_signal_dc1.setPen((pg.mkPen(100, 100, 255)))

        x = np.arange(data.shape[1])
        pos = data[0, :]
        if data[2, 1] > data[2, 0]:
            sig_dc0 = data[1, 1::2]
            sig_dc1 = data[1, 0::2]
        else:
            sig_dc1 = data[1, 1::2]
            sig_dc0 = data[1, 0::2]
        self.stream_curve.setData(x, pos)
        self.stream_signal_dc0.setData(x[::2], sig_dc0)
        self.stream_signal_dc1.setData(x[::2], sig_dc1)

    def _calculate_noise_floor(self, da):
        """ Calculate the noise level based on std dev of unpumped data."""
        self.avg_std = np.std(da.values[:len(da) // 20])
        label = 'Noise Floor:'
        label += '   {:15}:   {:.2E}'.format('Average', self.avg_std)
        self.noise_label.setText(label)

    def post_process(self, dataArray, filter=True, baseline=True, cutEdges=False):
        """ process curves before plotting"""
        da = dataArray.copy(deep=True)
        if self.butter_filter_checkbox.isChecked() and filter:
            da = self._apply_filter(da)
        if self.cb_remove_baseline.isChecked() and baseline:
            da = self._remove_baseline(da)
        if cutEdges:
            da = self._cut_edges(da)
        return da

    def _apply_filter(self, da):
        try:
            o, c = self.filter_order_spinbox.value(), self.filter_frequency_spinbox.value()
            b, a = butter(o, c)
            da.data = filtfilt(b, a, da)
            self.logger.debug('Applied {} order Butterworth filter, with cutoff {}'.format(o, c))
            return da
        except Exception as e:
            self.logger.debug('Failed applying Butterworth filter: {}'.format(e))
            return da

    def _remove_baseline(self, da):
        """ Set negative time delay signal to zero"""
        n_prepump = len(da) // 20
        res = da - da[:n_prepump].mean()
        return res

    def _cut_edges(self, da):
        """ Cut the side values of the trace to avoid artifacts in the average."""
        off = max(1, self.avg_side_cutoff.value())
        return da[off:-off]

    def _get_data_bounds(self):
        """ Determine the x an y ranges of data from the current curves."""
        try:
            for name, curve in self.curves.items():
                Xr_min = Yr_min = Xr_max = Yr_max = np.nan
                Xr, Yr = curve.dataBounds(0), curve.dataBounds(1)
                Xr_min = np.nanmin([Xr_min, Xr[0]])
                Xr_max = np.nanmax([Xr_max, Xr[1]])
                Yr_min = np.nanmin([Yr_min, Yr[0]])
                Yr_max = np.nanmax([Yr_max, Yr[1]])
            return (Xr_min, Xr_max), (Yr_min, Yr_max)

        except:
            return (np.nan, np.nan), (np.nan, np.nan)

    def lazy_autoscale(self, n):
        """ Autoscale based on last n datasets.

        Adds the current data range to a list and sets the range to the min-max
        of this list. The list is cut when exceeds n values.
        """
        xRange, yRange = self._get_data_bounds()
        if not np.nan in xRange:
            self._main_plot_ranges['Xr_min'].append(xRange[0])
            self._main_plot_ranges['Xr_max'].append(xRange[1])
        if not np.nan in yRange:
            self._main_plot_ranges['Yr_min'].append(yRange[0])
            self._main_plot_ranges['Yr_max'].append(yRange[1])

        if len(self._main_plot_ranges['Xr_min']) > n:
            for k, v in self._main_plot_ranges.items():
                _ = v.pop(0)
        try:
            xRange = np.nanmin(self._main_plot_ranges['Xr_min']), np.nanmax(self._main_plot_ranges['Xr_max'])
            yRange = np.nanmin(self._main_plot_ranges['Yr_min']), np.nanmax(self._main_plot_ranges['Yr_max'])
            self.main_plot.setRange(xRange=xRange, yRange=yRange)
        except:
            pass

    def set_autoscale(self):
        """ Turn on autoscale on main plot. """
        self.main_plot.enableAutoRange()
        self._scale_follow = False
        self.btn_autoscale.setEnabled(False)
        self.btn_follow.setEnabled(True)

    def set_showall(self):
        """ Fix view of main plot to show all data. """

        self._scale_follow = False
        self.main_plot.disableAutoRange()
        self.btn_autoscale.setEnabled(True)
        self.btn_follow.setEnabled(True)
        xrange, yrange = self._get_data_bounds()
        if not np.nan in xrange and not np.nan in yrange:
            self.main_plot.setRange(xRange=xrange, yRange=yrange)

    def set_follow(self):
        """ Set main plot ranges to follow data range. A slower version of autoscale."""
        self.main_plot.disableAutoRange()
        self._scale_follow = True
        self.btn_autoscale.setEnabled(True)
        self.btn_follow.setEnabled(False)


if __name__ == '__main__':
    pass
