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
from queue import Empty
import sys, os
import time
import traceback
import h5py
import numpy as np
import xarray as xr
from PyQt5 import QtCore

sys.path.append('D:\code\FemtoScan')
from instruments.cryostat import ITC503s as Cryostat

try:
    import nidaqmx
    from nidaqmx import stream_readers
    from nidaqmx.constants import Edge, AcquisitionType
except:
    print('no nidaqmx package found, only simulations available')

from scipy.optimize import curve_fit

from .misc import sech2_fwhm, sech2_fwhm_wings, sin, update_average, update_running_average, gaussian_fwhm, gaussian, transient_1expdec
from .misc import parse_setting, parse_category, write_setting, NoDataException

try:
    from fastscan.cscripts.project import project, project_r0

    print('Loaded Cython scripts')
except:
    print('WARNING: failed loading cython projector, loading python instead')
    from .cscripts.projectPy import project, project_r0

try:  # TODO: remove,
    sys.path.append(parse_setting('paths', 'instruments_repo'))
    from instruments.delaystage import DelayStage, Standa_8SMC5

    # from instruments.cryostat import Cryostat
    print('Loaded instruments')
except:
    print('WARNING: failed loading instruments repo')


# -----------------------------------------------------------------------------
#       thread management
# -----------------------------------------------------------------------------

class FastScanThreadManager(QtCore.QObject):
    """
    This class manages the streamer processor and fitter workers for the fast
    scan data acquisition and processing.

    """
    newStreamerData = QtCore.pyqtSignal(np.ndarray)
    newProcessedData = QtCore.pyqtSignal(xr.DataArray)
    newFitResult = QtCore.pyqtSignal(dict)
    newAverage = QtCore.pyqtSignal(xr.DataArray)
    # acquisitionStopped = QtCore.pyqtSignal()
    # finished = QtCore.pyqtSignal()
    # newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(Exception)

    acquisition_started = QtCore.pyqtSignal()
    acquisition_stopped = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger('{}.FastScanThreadManager'.format(__name__))
        self.logger.info('Created Thread Manager')

        # multiprocessing variables
        self._stream_queue = mp.Queue()  # Queue where to store unprocessed streamer data
        self._processed_queue = mp.Queue()  # Queue where to store projected data
        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(self.n_processors)

        # Data containers
        self.all_curves = None  # xarray containing all projected curves
        self.running_average = None  # xarray containing average of projected curves
        self.streamer_average = None  # np.array containing the average of streamer data #TODO: assert if this contains only last n averages as it should.
        self.n_streamer_averages = 0

        # running control parameters
        self._calculate_autocorrelation = None  # if true enables the autocorrelation fitting method to run
        self._recording_iteration = False  # for iterative temperature measurement
        self._max_avg_calc_time = 10  # maximum time (ms) it should take to calculate a new average
        self._skip_average = 0  # number of iterations to skipp average calculation
        self._should_stop = False  # when true stops the data acquisition as soon as possible
        self._streamer_running = False  # flag for chekcing weather the acquisition is running
        self._current_iteration = None  # keep track of which is the current iteration in the iterative measurement method.
        self._spos_fit_pars = None  # initialize the fit parameters for shaker position
        self._counter = 0  # Counter for thread safe wait function
        self._calc_avg_with_worker = parse_setting('fastscan', 'avg_use_worker')
        self._updating_avg = False
        # self.cryo = Cryostat(parse_setting('instruments', 'cryostat_com'))
        # self.delay_stage = DelayStage()

        # master clock which calls `on_timer`
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.master_clock)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()

        self._avg_timeout = 0

        # load instruments
        self.cryo = Cryostat(parse_setting('instruments', 'cryostat_com'))

        # self.create_streamer()

    # data processing threads

    def start_projector_worker(self, stream_data):
        """ Uses a thread to project stream data into pump-probe time scale.

        Launch a runnable thread from the pool to convert data from streamer
        into the correct time scale. When the projection is complete, a signal
        passes the result to self.on_projector_data method.

        Args:
            stream_data: np.array
                data acquired by streamer.
        """
        self.logger.debug('Starting projector')
        runnable = Runnable(projector,
                            stream_data=stream_data,
                            spos_fit_pars=self._spos_fit_pars,
                            use_dark_control=self.dark_control,
                            adc_step=self.shaker_position_step,
                            time_step=self.shaker_time_step,
                            use_r0=self.use_r0,
                            cut_range=self.cut_range
                            )
        self.pool.start(runnable)
        runnable.signals.result.connect(self.on_projector_data)


    def update_average_worker(self):
        new_projections = []
        for  i in range(self.processed_qsize):
            tmp = self._processed_queue.get()
            if tmp is not None:
                new_projections.append(tmp)
            else:
                break
        if len(new_projections) >0:
            self.logger.debug('Updating average with {} new projections'.format(len(new_projections)))
            self._updating_avg = True
            runnable = Runnable(update_running_average,new_projections,self.all_curves,self.n_averages)
            self.pool.start(runnable)
            runnable.signals.result.connect(self.on_updated_running_average)
        else:
            pass
            # self.logger.debug('no projections to work on....')

    def on_updated_running_average(self, tpl):
        """ Store and emit new average dataset, and start a new average calculation"""
        self.all_curves, self.running_average, dt= tpl
        self._updating_avg = False

        self.logger.debug('Updated average of {} curves in {} ms'.format(len(self.all_curves),dt*1000))
        self.newAverage.emit(self.running_average)
        if self._calculate_autocorrelation:
            self.start_autocorrelation_worker(self.running_average)
        # if not self._should_stop:
        #     self.update_average_worker()

    def start_autocorrelation_worker(self, da):
        """ Uses a thread to fit the autocorrelation function to the projected data.

        Args:
            da: xarray.DataArray
                data after projection processing.
        """
        runnable = Runnable(fit_autocorrelation_wings, da)#, expected_pulse_duration=.1)
        self.pool.start(runnable)
        runnable.signals.result.connect(self.on_fit_result)
        # runnable.signals.result.connect(self.newFitResult.emit)

    # Streamer management methods

    def create_streamer(self):
        """ Generate the streamer thread.

        This creates the thread which will acquire data from the ADC.
        """
        self.streamer_thread = QtCore.QThread()

        self.streamer = FastScanStreamer()
        self.streamer.newData[np.ndarray].connect(self.on_streamer_data)
        self.streamer.error.connect(self.error.emit)
        self.streamer.moveToThread(self.streamer_thread)
        self.streamer_thread.started.connect(self.streamer.start_acquisition)

    @QtCore.pyqtSlot()
    def start_streamer(self):
        """ Start the acquisition by starting the streamer thread"""
        self._should_stop = False
        self._streamer_running = True
        self.create_streamer()
        self.streamer_thread.start()
        self.acquisition_started.emit()
        self.logger.info('FastScanStreamer started')
        self.logger.debug('streamer settings: {}'.format(parse_category('fastscan')))

    @QtCore.pyqtSlot()
    def stop_streamer(self):
        """ Soft stop order for streamer.

        Streamer is set to stop at the next iteration.
        """
        self.logger.debug('\n\nFastScan Streamer is stopping.\n\n')
        self.streamer.stop_acquisition()
        self.acquisition_stopped.emit()
        self._should_stop = True

    # data handling pipeline

    @QtCore.pyqtSlot()
    def on_timer(self):
        """ Timer event.

        This is used for multiple purpouses:
         - to stop threads when they are supposed to
         - to end an acqusition which has finite number of iterations
         - to increase the _counter for 'wait' function
         - maybe other stuff too...
        """
        if self._updating_avg and self._avg_timer < self.averaging_timeout:
            self._avg_timer += self.master_clock
        else:
            self._avg_timer = 0
            self.update_average_worker()

        # Stop the streamer if button was pressed
        if self._should_stop:
            self.logger.debug(
                'Killing streamer: streamer queue: {} processed queue: {}'.format(self.stream_qsize,
                                                                                  self.processed_qsize))
            self.streamer_thread.exit()
            self._should_stop = False
            self._streamer_running = False
        # when streamer data is available, start a projector
        try:
            if not self._stream_queue.empty():
                _to_project = self._stream_queue.get()
                self.start_projector_worker(_to_project)
                self.logger.debug(
                    'Projecting an element from streamer queue: {} elements remaining'.format(self.stream_qsize))
        except Exception as e:
            self.logger.debug('Queue error: {}'.format(e))

        # manage iterative measurement. Start next step whien n_averages has been reached.
        if self._recording_iteration:
            try:
                if self.all_curves.shape[0] == self.n_averages:
                    self.logger.info('n of averages reached: ending iteration step')
                    self.end_iteration_step()
            except AttributeError as e:
                pass

        self._counter += 1  # increase the waiting counter

    @QtCore.pyqtSlot(np.ndarray)
    def on_streamer_data(self, streamer_data):
        """ Slot to handle streamer data.

        Performs the following actions in this order:

        1. Emit raw stramer data for plotting
        2. add stramer data to streamer queue, ready to be projected.
        3. update streamer average and keep count of how many we averaged

        Args:
            streamer_data: np.ndarray(N,M)
                array of dimension (N,M) where N is the number of different
                channels aquired, and M the number of samples (laser pulses)
                recorded at each trigger (shaker cycle)

        """
        # emit data for plotting
        self.newStreamerData.emit(streamer_data)

        # add data to stream, for projection
        self._stream_queue.put(streamer_data)
        self.logger.debug('Added data to stream queue, with shape {}'.format(streamer_data.shape))

        t0 = time.time()
        if self.streamer_average is None:
            self.streamer_average = streamer_data
            self.n_streamer_averages = 1
        else:
            self.n_streamer_averages += 1
            self.streamer_average = update_average(streamer_data, self.streamer_average, self.n_streamer_averages)
        self.logger.debug('{:.2f} ms| Streamer average updated ({} scans)'.format((time.time() - t0) * 1000,
                                                                                  self.n_streamer_averages))

    @QtCore.pyqtSlot(tuple)
    def on_projector_data(self, processed_dataarray_tuple):
        """ Slot to handle processed data.

        In input there comes a tuple containing an xarray of the projected data
        together with the fit parameters for the sine shape of the shaker position.

        This method then performs the following actions in this order:
        1. emit data for plotting.
        2. add data to the processed queue, ready for updating the runnning average
        3. if the counter _skip_average has reached 0, it recalculates the
        running average and emits it for plotting. Furthermore, measures how
        long calculating this took, and based on this time, it then defines how
        many cycles of this method it should wait (how many times it should get
        projected data - how many shaker cycles) before updating the average
        again. If the counter _skip_average has not reached 0, it decreases it by 1.
        4. If the autocorrelation is flagged true, it starts an autocorrelation
        fitting thread.

        """
        self.logger.debug('received processed data array')

        processed_dataarray, self._spos_fit_pars = processed_dataarray_tuple
        # send data to GUI for plotting "last curve"
        self.newProcessedData.emit(processed_dataarray)
        # add to queue for average calculation
        self._processed_queue.put(processed_dataarray)
        self.logger.debug('Added processed curve to queue')

        if not self._calc_avg_with_worker:

            if self._skip_average == 0:
                self.logger.debug('\nstarting average calculation: {} elements in queue'.format(self.processed_qsize))

                t0 = time.time()
                all_last_projected = []

                # take all elements available in processor queue and calculate the new running average.
                for i in range(self.processed_qsize):
                    try:
                        tmp = self._processed_queue.get(block=True, timeout=0.01).dropna('time')
                        if tmp is not None:
                            all_last_projected.append(tmp)  # drop values where no data was recorded (nans)
                    except Empty:  # bad method. one should NEVER catch queue empty errors... unreliable!!
                        self.logger.debug('queue reported empty. actual size: {}'.format(self.processed_qsize))
                        pass
                self.logger.debug('\n{} elements taken from the processed queue'.format(len(all_last_projected)))

                # if all_curves is not initialized yet, take one of the projected datasets and use it as first average
                if self.all_curves is None:
                    self.all_curves = processed_dataarray
                    self.running_average = all_last_projected.pop(0).dropna('time')

                # concatenate all (remaining) averages into the running average, using xarray methods for better aligning time axis
                n_left = len(all_last_projected)
                if n_left > 0:
                    self.all_curves = xr.concat([self.all_curves[-self.n_averages + n_left:], *all_last_projected], 'avg')
                    self.running_average = self.all_curves.mean('avg').dropna('time')

                self.newAverage.emit(self.running_average)

                # evaluate how long it took to update the average, and define how many cycles to skip based on the "maximum
                # processing time" defined in self._max_avg_calc_time. This should be much shorter than one cycle of the
                # shaker (1/10th)
                t_tot = (time.time() - t0) * 1000
                if t_tot > self._max_avg_calc_time:
                    self._skip_average = t_tot // self._max_avg_calc_time
                self.logger.debug('calculated average in {:.2f} ms'.format(t_tot))
            else:
                self.logger.debug(
                    'skipping average calculation, {} , queue size: {}'.format(-self._skip_average, self.processed_qsize))
                self._skip_average -= 1

            if self._calculate_autocorrelation:
                self.start_autocorrelation_worker(self.running_average)

    @QtCore.pyqtSlot(dict)
    def on_fit_result(self, fitDict):
        """ Slot to bounce the fit result signal."""
        self.newFitResult.emit(fitDict)

    # data I/O

    @QtCore.pyqtSlot()
    def reset_data(self):
        """ Reset the data in memory, by reinitializing all data containers.
        """
        # TODO: add popup check window
        self.running_average = None
        self.all_curves = None
        self.n_streamer_averages = None
        self.streamer_average = None
        for i in range(self.processed_qsize):
            self._processed_queue.get()
        for i in range(self.stream_qsize):
            self._stream_queue.get()
        # while not self._stream_queue.empty():
        #     self._stream_queue.get()
        # while not self._processed_queue.empty():

    def save_data(self, filename, all_data=True):
        """ Save data contained in memory.

        Save average curves and optionally the single traces for each shaker
        period. Additionally collects all available metadata and settings and
        stores all in an HDF5 container.

        Args:
            filename: path
                path and file name of the generated h5 file. Adds .h5 extension
                if missing.
            all_data:
                if True, saves all projected data for each shaker loop measured,
                otherwise only saves the avereage curve.
        Returns:

        """
        if not '.h5' in filename:
            filename += '.h5'

        if self.streamer_average is not None:
            directory = os.path.dirname(filename) #TODO: fix folder not found error
            if not os.path.isdir(directory):
                os.mkdir(directory)

            with h5py.File(filename, 'w') as f:

                f.create_dataset('/raw/avg', data=self.streamer_average)
                if all_data:
                    f.create_dataset('/all_data/data', data=self.all_curves.values)
                    f.create_dataset('/all_data/time_axis', data=self.all_curves.time)
                f.create_dataset('/avg/data', data=self.running_average.values)
                f.create_dataset('/avg/time_axis', data=self.running_average.time)

                for k, v in parse_category('fastscan').items():
                    if isinstance(v, bool):
                        f.create_dataset('/settings/{}'.format(k), data=v, dtype=bool)
                    else:
                        f.create_dataset('/settings/{}'.format(k), data=v)


        else:
            self.logger.info('no data to save yet...')
            # f.create_group('/settings')

    # shaker calibration

    def calibrate_shaker(self, iterations, integration):
        """
        Shaker calibration method.

        TODO: add description of shakercalib method.
        Args:
            iterations:
                number of full time scales to acquire.
            integration:
                number of shaker cycles to integrate on for each iteration.
        Returns:
            plots the result of the calibration. Prints output result.

        """
        assert not self._streamer_running, 'Cannot run Shaker calibration while streamer is running'

        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        # self.start_streamer()
        write_setting(0, 'fastscan - simulation', 'center_position')

        self.create_streamer()

        # stream = self.streamer.simulate_single_shot(integration)
        stream = self.streamer.measure_single_shot(integration)

        projected = project(stream, self.dark_control,
                            self.shaker_position_step, 0.05)
        min_ = projected.time.min()
        max_ = projected.time.max()
        print('\n - ')
        calib_positions_ = np.linspace(min_ * .7, max_ * .7, iterations // 2)
        calib_positions = np.concatenate((calib_positions_, calib_positions_[::-1]))
        centers = []
        # np.random.shuffle(calib_positions)
        for pos in calib_positions:
            print('\n - moving stage')

            self.delay_stage.move_absolute(pos)
            if parse_setting('fastscan', 'simulate'):
                write_setting(pos, 'fastscan - simulation', 'center_position')

            # stream = self.streamer.simulate_single_shot(integration)
            stream = self.streamer.measure_single_shot(integration)
            projected = project(stream, self.dark_control,
                                self.shaker_position_step, 1)
            res = fit_autocorrelation(projected)
            print('\n - fitted, result it: {}'.format(res['popt'][1]))
            centers.append(res['popt'][1])
            # plt.plot(projected)
        # plt.show()

        steps = []
        calib_steps = []
        print('\n - calculating shift')

        for i in range(len(centers) - 1):
            dy = np.abs(centers[i] - centers[i + 1])
            dx = np.abs(calib_positions[i] - calib_positions[i + 1])
            if dx != 0:
                steps.append(dy / dx)

        mean = np.mean(steps)
        std = np.std(steps)
        good_steps = [x for x in steps if (np.abs(x - mean) < 2 * std)]

        # correction_factor = np.mean(calib_steps)/np.mean(steps)

        # write_setting('fastscan', 'shaker_ps_per_step', pos)
        print('\n\n Shaker Calibration result: {} or {}'.format(1. / np.mean(steps), 1. / np.mean(good_steps)))

        def lin(x, a, b):
            return a * x + b

        # plt.plot(calib_positions, centers, 'ob')

        # try:
        popt_, pcov_ = curve_fit(lin, centers, calib_positions)

        cpos_fit = lin(np.array(centers), *popt_)
        calib_pos_good = []
        centers_good = []
        for i in range(len(centers)):
            std = np.std(calib_positions - cpos_fit)
            if np.abs(calib_positions[i] - cpos_fit[i]) < 1.2 * std:
                calib_pos_good.append(calib_positions[i])
                centers_good.append(centers[i])
        popt, pcov = curve_fit(lin, centers_good, calib_pos_good)

        plt.plot(centers, calib_positions, 'o')
        plt.plot(centers_good, calib_pos_good, '.')
        x = np.linspace(min(centers), max(centers), 100)
        plt.plot(x, lin(x, *popt))
        # except:
        #     pass
        print('\n\n Shaker Calibration result:  {} | {}'.format(np.mean(good_steps), popt[0]))

        plt.show()

    # iterative temperature measurement

    def start_iterative_measurement(self, temperatures, savename):
        """ Starts a temperature dependence scan series.

        Args:
            temperatures: list of float
                list of temperatures at which to perform measurements.
            savename:
                base name of the save file (and path) to which temperature info
                will be appended. ex: c:\path\to\scan_folder\filename_T004,3K.h5
        """
        assert True
        self.temperatures = temperatures
        self.iterative_measurement_name = savename
        # self.start_streamer()
        # self.wait(1000)
        # self.save_data(savename) #TODO: remove
        # self.stop_streamer()
        # self.reset_data()
        self.logger.info('starting measurement loop')
        self._current_iteration = 0
        # self.stop_streamer()
        self.start_streamer()
        time.sleep(2)
        self.start_next_iteration()

    @QtCore.pyqtSlot()
    def start_next_iteration(self):
        """
        Initialize the next measurement iteration in the temperature dependence
        scan series.
        """
        self.stop_streamer()

        if self._current_iteration >= len(self.temperatures):
            self._current_iteration = None
            print('\n\n\n\nMEASUREMENT FINISHED\n\n\n')
            self.logger.info('Iterative mesasurement complete!!')
        else:
            self.cryo.connect()
            self.logger.info('Connected to Cryostat: setting temperature....')
            self.cryo.set_temperature(self.temperatures[self._current_iteration])
            runnable = Runnable(self.cryo.check_temp, tolerance=.1, sleep_time=1)
            self.pool.start(runnable)
            runnable.signals.finished.connect(self.measure_current_iteration)

    @QtCore.pyqtSlot()
    def measure_current_iteration(self):
        """ Start the acquisition for the current measurement iteration."""
        self.cryo.disconnect()
        self.logger.info('Temperature stable, measuring interation {}, {}K'.format(self._current_iteration,
                                                                                   self.temperatures[
                                                                                       self._current_iteration]))
        self.stop_streamer()
        self.reset_data()
        self.start_streamer()
        self._recording_iteration = True

    @QtCore.pyqtSlot()
    def end_iteration_step(self):
        """ Complete and finalize the current measurement iteration."""
        self.logger.info('Stopping iteration')
        self._recording_iteration = False

        t = self.temperatures[self._current_iteration]
        temp_string = '_{:0.2f}K'.format(float(t)).replace('.', ',')

        savename = self.iterative_measurement_name + temp_string
        self.logger.info('Iteration {} complete. Saved data as {}'.format(self._current_iteration, savename))
        print(savename)
        self.save_data(savename)

        self.stop_streamer()
        self._current_iteration += 1
        self.start_next_iteration()

    @staticmethod
    def check_temperature_stability(cryo, tolerance=.2, sleep_time=.1):
        """ Tests the sample temperature stability. """
        temp = []
        diff = 100000.
        while diff > tolerance:
            time.sleep(sleep_time)
            temp.append(cryo.get_temperature())
            if len(temp) > 10:
                temp.pop(0)
                diff = max([abs(x - cryo.temperature_target) for x in temp])
                print(f'cryo stabilizing: delta={diff}')

    def wait(self, n, timeout=1000):
        """ Gui safe waiting function.

        Args:
            n: int
                number of clock cycles to wait
            timeout: int
                number of clock cycles after which the waiting will be terminated
                notwithstanding n.
        """
        self._counter = 0
        i = 0
        while self._counter < n:
            i += 1
            if i > timeout:
                break

    @QtCore.pyqtSlot()
    def close(self):
        """ stop the streamer when closing this widget."""
        self.stop_streamer()

    ### Properties

    @property
    def stream_qsize(self):
        """ State of dark control. If True it's on."""
        try:
            val = self._stream_queue.qsize()
        except:
            val = -1
        return val

    @property
    def processed_qsize(self):
        """ size of the processed queue"""
        try:
            val = self._processed_queue.qsize()
        except:
            val = -1
        return val

    @property
    def dark_control(self):
        """ State of dark control. If True it's on."""
        return parse_setting('fastscan', 'dark_control')

    @dark_control.setter
    def dark_control(self, val):
        assert isinstance(val, bool), 'dark control must be boolean.'
        write_setting(val, 'fastscan', 'dark_control')

    @property
    def averaging_timeout(self):
        """ number of clock cycles to wait before forcefully computing the average."""
        return parse_setting('fastscan', 'averaging_timeout')

    @averaging_timeout.setter
    def averaging_timeout(self, val):
        assert isinstance(val, int), 'averaging_timeout must be integ er.'
        write_setting(val, 'fastscan', 'averaging_timeout')

    @property
    def master_clock(self):
        """ set the master clock frequency."""
        return parse_setting('fastscan', 'master_clock')

    @master_clock.setter
    def master_clock(self, val):
        assert isinstance(val, int), 'average_timeout must be integer.'
        write_setting(val, 'fastscan', 'master_clock')

    @property
    def use_r0(self):
        """ Choose to output DR/R or DR. If True it's DR/R."""
        return parse_setting('fastscan', 'use_r0')

    @property
    def cut_range(self):
        """ Choose to output DR/R or DR. If True it's DR/R."""
        return parse_setting('fastscan', 'cut_range')
    @use_r0.setter
    def use_r0(self, val):
        assert isinstance(val, bool), 'use_r0 must be boolean.'
        write_setting(val, 'fastscan', 'use_r0')

    @property
    def n_processors(self):
        """ Number of processors to use for workers."""
        return parse_setting('fastscan', 'n_processors')

    @n_processors.setter
    def n_processors(self, val):
        assert isinstance(val, int), 'dark control must be boolean.'
        assert val < os.cpu_count(), 'Too many processors, cant be more than cpu count: {}'.format(os.cpu_count())
        write_setting(val, 'fastscan', 'n_processors')
        self.create_processors()

    @property
    def n_averages(self):
        """ Number of averages to keep in the running average memory."""
        return parse_setting('fastscan', 'n_averages')

    @n_averages.setter
    def n_averages(self, val):
        assert val > 0, 'cannot set below 1'
        write_setting(val, 'fastscan', 'n_averages')
        self._n_averages = val
        self.logger.debug('n_averages set to {}'.format(val))

    @property
    def n_samples(self):
        """ Number of laser pulses to measure at each acquisition trigger pulse. """
        return parse_setting('fastscan', 'n_samples')

    @n_samples.setter
    def n_samples(self, val):
        assert val > 0, 'cannot set below 1'
        write_setting(val, 'fastscan', 'n_samples')
        self.logger.debug('n_samples set to {}'.format(val))

    @property
    def shaker_gain(self):
        """ Shaker position gain as in ScanDelay software."""

        return parse_setting('fastscan', 'shaker_gain')

    @shaker_gain.setter
    def shaker_gain(self, val):
        if isinstance(val, str): val = int(val)
        assert val in [1, 10, 100], 'gain can be 1,10,100 only'
        write_setting(val, 'fastscan', 'shaker_gain')
        self.logger.debug('n_samples set to {}'.format(val))

    @property
    def shaker_position_step(self):
        """ Shaker position ADC step size in v"""
        shaker_position_step =  parse_setting('fastscan', 'shaker_position_step')
        # shaker_scaling_factor =  parse_setting('fastscan', 'shaker_scaling_factor')
        return shaker_position_step * self.shaker_scaling_factor

    @property
    def shaker_ps_per_step(self):
        """ Shaker position ADC step size in ps"""
        shaker_ps_per_step =  parse_setting('fastscan', 'shaker_ps_per_step')
        # shaker_scaling_factor =  parse_setting('fastscan', 'shaker_scaling_factor')
        return shaker_ps_per_step * self.shaker_scaling_factor

    @property
    def shaker_scaling_factor(self):
        """ Shaker resampling factor. Binns together n steps to reduce the number
        of time points measured, reducing noise on each point by sqrt(n)."""
        return parse_setting('fastscan', 'shaker_scaling_factor')

    @shaker_scaling_factor.setter
    def shaker_scaling_factor(self, val):
        assert val > 0, 'Scaling factor should be positive.'
        assert isinstance(val,int), 'Scaling factor should be an integer.'

        write_setting(val, 'fastscan', 'shaker_scaling_factor')
        self.logger.debug('shaker_scaling_factor set to {}'.format(val))

    @property
    def shaker_time_step(self):
        """ Shaker digital step in ps.

        Takes into account ADC conversion and shaker gain to return the minimum
        step between two points converted from the shaker analog position signal.
        This also defines the time resolution of the measurement.

        Returns: float

        """
        return self.shaker_ps_per_step / self.shaker_gain

    @property
    def stage_position(self):
        """ Position of the probe delay stage."""
        return self.delay_stage.position_get()

    @stage_position.setter
    def stage_position(self, val):
        self.delay_stage.move_absolute(val)


class RunnableSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    """
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)


class Runnable(QtCore.QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals
    and wrap-up.
    :param callback: The function callback to run on this worker
    :thread. Supplied args and
    kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    :
    '''
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, fn, *args, **kwargs):
        super(Runnable, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = RunnableSignals()
        # Add the callback to our kwargs
        # kwargs['progress_callback'] = self.signals.progress

    @QtCore.pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


def fit_autocorrelation_wings(da, prev_result=None):
    """ fits the given data to a sech2 pulse shape"""
    da_ = da.dropna('time')

    n_wings = parse_setting('autocorrelation guess', 'n_wings')
    fit_curve_points = parse_setting('autocorrelation guess', 'fit_curve_points')
    fit_curve_sigmas = parse_setting('autocorrelation guess', 'fit_curve_sigmas')

    if prev_result is None or np.nan in prev_result['popt']:
        xc = da_.time[np.argmax(da_.values)]
        off = da_[da_.time - xc > .2].mean()
        a = da_.max() - off
        fwhm = parse_setting('autocorrelation guess', 'fwhm')
        wing_sep = parse_setting('autocorrelation guess', 'wing_sep')
        wing_ratio = parse_setting('autocorrelation guess', 'wing_ratio')
        guess = [a, xc, fwhm, off, wing_sep, wing_ratio]
        # bounds = [[0, xc - 1, 0.001, -np.inf, 0.00001, 0.000001],
        #           [100, xc + 1, 10, np.inf, 10, 10]]
    else:
        guess = prev_result['popt']

    def sech_wings(x, a, xc, fwhm, off, wing_sep, wing_ratio):
        return sech2_fwhm_wings(x, a, xc, fwhm, off, wing_sep, wing_ratio, n_wings)

    try:
        popt, pcov = curve_fit(sech_wings, da_.time, da_, p0=guess)  # , bounds=bounds)

        perr = np.sqrt(np.diag(pcov))
        x = np.linspace(popt[1] - fit_curve_sigmas * popt[2], popt[1] + fit_curve_sigmas * popt[2], fit_curve_points)
        curve = xr.DataArray(sech_wings(x, *popt), coords={'time': x}, dims='time')
    except RuntimeError:
        popt = pcov = perr = curve = None
    fitDict = {'popt': popt,
               'pcov': pcov,
               'perr': perr,
               'curve': curve,
               }
    return fitDict


def fit_autocorrelation(da, expected_pulse_duration=.1):
    """ fits the given data to a sech2 pulse shape"""
    da_ = da.dropna('time')

    xc = da_.time[np.argmax(da_.values)]
    off = da_[da_.time - xc > .2].mean()
    a = da_.max() - off

    guess = [a, xc, expected_pulse_duration, off]
    try:
        popt, pcov = curve_fit(sech2_fwhm, da_.time, da_, p0=guess)
    except RuntimeError:
        popt, pcov = [0, 0, 0, 0], np.zeros((4, 4))
    fitDict = {'popt': popt,
               'pcov': pcov,
               'perr': np.sqrt(np.diag(pcov)),
               'curve': xr.DataArray(sech2_fwhm(da_.time, *popt), coords={'time': da_.time}, dims='time')
               }
    return fitDict


def projector(stream_data, spos_fit_pars=None, use_dark_control=True, adc_step=0.000152587890625, time_step=.05,
              use_r0=True, cut_range=False):
    """

    Args:
        stream_data: ndarray
            streamer data, shape should be: (number of channels, number of samples)
        **kwargs:
            :method: str | fast
                projection method. Can be sine_fit, fast
            :use_dark_control: bool | False
                If to use the 3rd channel as dark control information
            :use_r0: bool | False
                if to use 4th channel as r0 (static reflectivity from reference channel)
                for obtaining dR/R
    Returns:

    """
    assert isinstance(stream_data, np.ndarray)
    t0 = time.time()
    logger = logging.getLogger('{}.Projector'.format(__name__))

    if stream_data.shape[0] == 4 and use_r0:  # calculate dR/R only when required and when data for R0 is there
        use_r0 = True
    else:
        use_r0 = False
    spos_analog = stream_data[0]
    signal = stream_data[1]
    dark_control = stream_data[2]

    x = np.arange(0, stream_data.shape[1], 1)

    if spos_fit_pars is None: #TODO: generalise fitting parameters. this might break on new setups
        g_amp = spos_analog.max() - spos_analog.min()
        g_freq = 15000 / np.pi
        g_phase = 0
        g_offset = g_amp / 5
        guess = [g_amp, g_freq, g_phase, g_offset]
    else:
        guess = spos_fit_pars
    popt, pcov = curve_fit(sin, x, spos_analog, p0=guess)
    spos_fit_pars = popt
    spos = np.array(sin(x, *popt) / adc_step, dtype=int)

    if cut_range:
        lims = [np.argmin(spos),np.argmax(spos)]
        f = min(lims)
        t = max(lims)
    else:
        f,t = 0,-1

    spos = spos[f:t]
    signal = signal[f:t]
    dark_control = dark_control[f:t]

    if use_r0:
        reference = stream_data[3][f:t]
        result = project_r0(spos, signal, dark_control, reference, use_dark_control)
    else:
        result = project(spos, signal, dark_control, use_dark_control)

    time_axis = np.arange(spos.min(), spos.max() + 1, 1) * time_step
    output = xr.DataArray(result, coords={'time': time_axis}, dims='time').dropna('time')
    logger.debug(
        '{:.2f} ms | projecting time r0:{}. dc:{} '.format((time.time() - t0) * 1000, use_r0, use_dark_control))

    return (output, spos_fit_pars)


# -----------------------------------------------------------------------------
#       Streamer
# -----------------------------------------------------------------------------

class FastScanStreamer(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    newData = QtCore.pyqtSignal(np.ndarray)
    error = QtCore.pyqtSignal(Exception)
    niChannel_order = ['shaker_position', 'signal', 'dark_control', 'reference']

    def __init__(self, ):
        super().__init__()
        self.logger = logging.getLogger('{}.FastScanStreamer'.format(__name__))
        self.logger.info('Created FastScanStreamer')

        self.init_ni_channels()

        self.should_stop = True  # when true it does not start a new measurement iteration.

    def init_ni_channels(self):  # TODO: choose channels from settings

        for k, v in parse_category('fastscan').items():
            setattr(self, k, v)

        self.niChannels = {  # default channels
            'shaker_position': "Dev1/ai0",
            'signal': "Dev1/ai1",
            'dark_control': "Dev1/ai2",
            'reference': "Dev1/ai3"}

        self.niTriggers = {  # default triggers
            'shaker_trigger': "/Dev1/PFI1",
            'laser_trigger': "/Dev1/PFI0"}
        #        try:
        #            self.niChannels = {}
        #            for k, v in parse_category('ni_signal_channels').items():
        #                self.niChannels[k] = v
        #        except:
        #            self.logger.critical('failed reading signal from SETTINGS, using default channels.')
        #        try:
        #            self.niTriggers = {}
        #            for k, v in parse_category('ni_trigger_channels').items():
        #                self.niTriggers[k] = v
        #        except:
        #            self.logger.critical('failed reading trigger channels from SETTINGS, using default channels.')

        self.data = np.zeros((len(self.niChannels), self.n_samples))

    @QtCore.pyqtSlot()
    def start_acquisition(self):
        if self.simulate:
            self.logger.info('Started streamer simulation in {} mode'.format(self.acquisition_mode))
            self.measure_simulated()
        else:
            if self.acquisition_mode == 'continuous':
                self.logger.info('Started NI continuous Streamer ')
                self.measure_continuous()

            elif self.acquisition_mode == 'triggered':
                self.logger.info('Started NI triggered Streamer ')
                self.measure_triggered()

    @QtCore.pyqtSlot()
    def stop_acquisition(self):
        self.logger.info('Streamer ordered to stop...')
        self.should_stop = True

    def measure_continuous(self):
        try:
            with nidaqmx.Task() as task:

                self.reader = stream_readers.AnalogMultiChannelReader(task.in_stream)
                self.logger.debug('reader initialized')
                task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  # shaker position chanel
                task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  # signal chanel
                task.ai_channels.add_ai_voltage_chan("Dev1/ai2")  # dark control chanel
                task.ai_channels.add_ai_voltage_chan("Dev1/ai3")  # reference

                self.logger.debug('added 3 tasks')
                task.timing.cfg_samp_clk_timing(1000000, source="/Dev1/PFI0",
                                                active_edge=Edge.RISING,
                                                sample_mode=AcquisitionType.CONTINUOUS)
                task.start()
                self.logger.info('FastScanStreamer taks started')

                self.should_stop = False
                i = 0
                while not self.should_stop:
                    i += 1
                    self.logger.debug('measuring cycle {}'.format(i))
                    self.reader.read_many_sample(self.data, number_of_samples_per_channel=self.n_samples)
                    self.logger.debug('Recieved data from NI card: mean axis 0 = {}'.format(self.data[0].mean()))
                    self.newData.emit(self.data)

                self.logger.warning('Acquisition stopped.')
                self.finished.emit()

        except Exception as e:
            self.logger.warning('Error while starting streamer: \n{}'.format(e))
            self.error.emit(e)

    def measure_triggered(self):
        """ Define tasks and triggers for NIcard.

        At each trigger signal from the shaker, it reads a number of samples
        (self.n_samples) triggered by the laser pulses. It records the channels
        given in self.niChannels.
        The data is then emitted by the newData signal, and will have the shape
        (number of channels, number of samples).
        """
        try:
            with nidaqmx.Task() as task:
                loaded_channels = 0
                for chan in self.niChannel_order:
                    try:
                        task.ai_channels.add_ai_voltage_chan(self.niChannels[chan])
                        loaded_channels += 1
                    except KeyError:
                        self.logger.critical('No {} channel found'.format(chan))
                    except:
                        self.logger.critical('Failed connecting to {} channel @ {}'.format(chan, self.niChannels[chan]))

                self.logger.debug('added {} tasks'.format(loaded_channels))
                task.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source=self.niTriggers['shaker_trigger'],
                                                                    trigger_edge=Edge.RISING)
                task.timing.cfg_samp_clk_timing(100000, samps_per_chan=self.n_samples,
                                                source=self.niTriggers['laser_trigger'],
                                                active_edge=Edge.RISING,
                                                sample_mode=AcquisitionType.FINITE)  # external clock chanel

                self.should_stop = False
                i = 0
                while not self.should_stop:
                    i += 1
                    self.logger.debug('measuring cycle {}'.format(i))
                    self.data = np.array(task.read(number_of_samples_per_channel=self.n_samples))
                    self.newData.emit(self.data)

                self.logger.warning('Acquisition stopped.')
                self.finished.emit()
        except Exception as e:
            self.logger.warning('Error while starting streamer: \n{}'.format(e))
            self.error.emit(e)

    def measure_single_shot(self, n):
        try:
            with nidaqmx.Task() as task:
                for k, v in self.niChannels:  # add all channels to be recorded
                    task.ai_channels.add_ai_voltage_chan(v)
                self.logger.debug('added {} tasks'.format(len(self.niChannels)))

                task.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1",
                                                                    trigger_edge=Edge.RISING)
                task.timing.cfg_samp_clk_timing(100000, samps_per_chan=self.n_samples,
                                                source="/Dev1/PFI0",
                                                active_edge=Edge.RISING,
                                                sample_mode=AcquisitionType.FINITE)  # external clock chanel

                self.should_stop = False
                i = 0
                data = np.zeros((n, *self.data.shape))
                for i in range(n):
                    self.logger.debug('measuring cycle {}'.format(i))
                    data[i, ...] = np.array(task.read(number_of_samples_per_channel=self.n_samples))
                return (data.mean(axis=0))

        except Exception as e:
            self.logger.warning('Error while starting streamer: \n{}'.format(e))
            self.error.emit(e)

    def simulate_single_shot(self, n):
        self.should_stop = False
        i = 0
        sim_parameters = parse_category('fastscan - simulation')
        fit_parameters = [sim_parameters['amplitude'],
                          sim_parameters['center_position'],
                          sim_parameters['fwhm'],
                          sim_parameters['offset']
                          ]
        step = parse_setting('fastscan', 'shaker_position_step')
        ps_per_step = parse_setting('fastscan', 'shaker_ps_per_step')  # ADC step size - corresponds to 25fs
        ps_per_step *= parse_setting('fastscan', 'shaker_gain')  # correct for shaker gain factor

        data = np.zeros((n, *self.data.shape))
        for i in range(n):
            self.logger.debug('measuring cycle {}'.format(i))
            data[i, ...] = simulate_measure(self.data,
                                            function=sim_parameters['function'],
                                            args=fit_parameters,
                                            amplitude=sim_parameters['shaker_amplitude'],
                                            mode=self.acquisition_mode,
                                            step=step,
                                            ps_per_step=ps_per_step,
                                            )

        return (data.mean(axis=0))

    def measure_simulated(self):
        self.should_stop = False
        i = 0
        sim_parameters = parse_category('fastscan - simulation')
        fit_parameters = [sim_parameters['amplitude'],
                          sim_parameters['center_position'],
                          sim_parameters['fwhm'],
                          sim_parameters['offset']
                          ]
        if sim_parameters['function'] == 'sech2_fwhm_wings':
            fit_parameters.append(sim_parameters['wing_sep'] * sim_parameters['fwhm'])
            fit_parameters.append(sim_parameters['wing_ratio'])
            fit_parameters.append(sim_parameters['n_wings'])

        step = parse_setting('fastscan', 'shaker_position_step')
        ps_per_step = parse_setting('fastscan', 'shaker_ps_per_step')  # ADC step size - corresponds to 25fs
        ps_per_step *= parse_setting('fastscan', 'shaker_gain')  # correct for shaker gain factor

        while not self.should_stop:
            i += 1
            self.logger.debug('simulating measurement cycle #{}'.format(i))
            t0 = time.time()

            self.data = simulate_measure(self.data,
                                         function=sim_parameters['function'],
                                         args=fit_parameters,
                                         amplitude=sim_parameters['shaker_amplitude'],
                                         mode=self.acquisition_mode,
                                         step=step,
                                         ps_per_step=ps_per_step,
                                         )
            dt = time.time() - t0
            time.sleep(max(self.n_samples / 273000 - dt, 0))
            self.newData.emit(self.data)
            self.logger.debug(
                'simulated data in {:.2f} ms - real would take {:.2f} - '
                'outputting array of shape {}'.format(dt * 1000,
                                                      self.n_samples / 273,
                                                      self.data.shape))


def simulate_measure(data, function='sech2_fwhm', args=[.5, -2, .085, 1],
                     amplitude=10, mode='triggered',
                     step=0.000152587890625, ps_per_step=.05):
    args_ = args[:]

    if function == 'gauss_fwhm':
        f = gaussian_fwhm
        args_[1] *= step / ps_per_step  # transform ps to voltage
        args_[2] *= step / ps_per_step  # transform ps to voltage
    elif function == 'gaussian':
        f = gaussian
        args_[1] *= step / ps_per_step  # transform ps to voltage
        args_[2] *= step / ps_per_step  # transform ps to voltage
        args_.pop(0)
        args_.pop(-1)
    elif function == 'sech2_fwhm':
        f = sech2_fwhm
        args_[1] *= step / ps_per_step  # transform ps to voltage
        args_[2] *= step / ps_per_step  # transform ps to voltage
    elif function == 'sech2_fwhm_wings':
        f = sech2_fwhm_wings
        args_[1] *= step / ps_per_step  # transform ps to voltage
        args_[2] *= step / ps_per_step  # transform ps to voltage
    elif function == 'transient_1expdec':
        f = transient_1expdec
        args_ = [2, 20, 1, 1, .01, -10]
        args_[1] *= step / ps_per_step  # transform ps to voltage
        args_[2] *= step / ps_per_step  # transform ps to voltage
        args_[5] *= step / ps_per_step  # transform ps to voltage
    else:
        raise NotImplementedError('no funcion called {}, please use gauss or sech2'.format(function))
    #########
    n = np.arange(len(data[0]))
    noise = np.random.rand(len(n))
    if mode == 'continuous':
        phase = noise[0] * 2 * np.pi
    else:
        phase = 0
    amplitude = amplitude * step / ps_per_step * (1 + .02 * np.random.uniform(-1, 1))

    data[0, :] = np.cos(2 * np.pi * n / 30000 + phase) * amplitude / 2  # in volt

    data[1, 1::2] = data[0, 1::2] / 3
    data[1, ::2] = f(data[0, ::2], *args_) + noise[::2] + data[0, ::2] / 3
    data[2, ::2] = True
    data[2, 1::2] = False
    if data.shape[0] == 4:
        data[3, :] = 2

    return data


if __name__ == '__main__':
    pass
