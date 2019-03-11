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

import pyqtgraph as pg
import xarray as xr
import numpy as np
import multiprocessing as mp
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from pyqtgraph.Qt import QtCore, QtGui
import logging

class FastScanPlotWidget(QWidget):

    def __init__(self):
        super(FastScanPlotWidget, self).__init__()
        self.logger = logging.getLogger('-.{}.PlotWidget'.format(__name__))
        self.logger.info('Created PlotWidget')
        self.main_plot_lines = {}
        self.secondary_plot_lines = {}
        self.clock = QTimer()
        self.clock.setInterval(1000./30)
        self.clock.timeout.connect(self.on_clock)
        self.clock.start()

        self.plot_queue = mp.Queue()

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

        self.da = None
        self.last_curve = None
        self.add_main_plot_line('last_curve',(255,255,255))
        self.average_curve = None
        self.add_main_plot_line('average_curve',(100,255,100))

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

    @QtCore.pyqtSlot(dict)
    def update_dataset(self,data_dict):
        self.plot_queue.put(data_dict)
        # if 'all' in dataset.data_vars:
        #     ds = dataset['all'][-1,...]
        #     self.plot_main('last',x=np.array(ds),y=np.array(ds.time),pen=(255,255,255))
        # if 'average' in dataset.data_vars:
        #     ds = dataset['average'][-1,...]
        #     self.plot_main('last',x=np.array(ds),y=np.array(ds.time),pen=(255,255,255))

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

    def plot_main(self, name, x=None, y=None,pen=(255,255,255)):
        if name not in self.main_plot_lines.keys():
            self.logger.debug('adding {} curve to main plot'.format(name))
            self.add_main_plot_line(name, pen)
        self.main_plot_lines[name]['x'] = x
        self.main_plot_lines[name]['y'] = y

    def plot_secondary(self, name, x=None, y=None,pen=(255,255,255)):
        if name not in self.secondary_plot_lines.keys():
            self.logger.debug('adding {} curve to secondary plot'.format(name))
            self.add_secondary_plot_line(name, pen)
        self.secondary_plot_lines[name]['x'] = x
        self.secondary_plot_lines[name]['y'] = y

    def plot_main_xr(self, name, dataArray,pen=(255,255,255)):
        if name not in self.main_plot_lines.keys():
            self.logger.debug('adding {} curve to main plot'.format(name))
            self.add_main_plot_line(name, pen)

        self.main_plot_lines[name]['x'] = dataArray.time
        self.main_plot_lines[name]['y'] = dataArray
        self.main_plot_lines[name]['plot'].setData(dataArray.time,dataArray)

    def plot_secondary_xr(self, name, dataArray,pen=(255,255,255)):
        if name not in self.secondary_plot_lines.keys():
            self.logger.debug('adding {} curve to secondary plot'.format(name))
            self.add_secondary_plot_line(name, pen)

        self.secondary_plot_lines[name]['x'] = dataArray.time
        self.secondary_plot_lines[name]['y'] = dataArray

    def on_clock(self):
        if not self.plot_queue.empty():
            newData = self.plot_queue.get()
            print('plotting')
            for key, array in newData.items():
                if key == 'all':
                    if 'avg' in array.dims:
                        self.plot_main_xr('last',array[-1])
                else:
                    self.plot_main_xr(key,array,pen=(0,255,0))


        # try:
        #     for key, val in self.main_plot_lines.items():
        #         val['plot'].setData(val['x'], val['y'])
        #     for key, val in self.secondary_plot_lines.items():
        #         val['plot'].setData(val['x'], val['y'])
        # except KeyError:
        #     pass
        # except Exception as e:
        #     pass
        #     self.logger.warning('Failed Plotting. \nData:{} - {}\n Error: {}'.format(self.main_plot_lines.keys(),self.secondary_plot_lines.keys(),e))

def main():
    pass


if __name__ == '__main__':
    main()
