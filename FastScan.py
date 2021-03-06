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
#
# import nidaqmx
# import numpy as np
# import pyqtgraph as pg
# from PyQt5.pQtCore import QCoreApplication, QTimer
# from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, \
#     QGroupBox
# from nidaqmx.constants import Edge, AcquisitionType


import logging
import sys, os
from logging.config import fileConfig

from PyQt5 import QtCore, QtWidgets, QtGui

from fastscan.gui import FastScanMainWindow
# from utilities.settings import set_default_settings


def main():
    from fastscan.misc import my_exception_hook
    # used to see errors generated by PyQt5 in pycharm:
    sys._excepthook = sys.excepthook
    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook

    # create main logger
    # fileConfig('./cfg/logging_config.ini')
    fileConfig('./SETTINGS.ini')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug('Started logger')

    app = QtCore.QCoreApplication.instance()
    app_icon = QtGui.QIcon()
    # app_icon.addFile('icons/logo16.png', QtCore.QSize(16, 16))
    # app_icon.addFile('icons/logo24.png', QtCore.QSize(24, 24))
    # app_icon.addFile('icons/logo32.png', QtCore.QSize(32, 32))
    # app_icon.addFile('icons/logo48.png', QtCore.QSize(48, 48))
    # app_icon.addFile('icons/logo256.png', QtCore.QSize(256, 256))
    # app.setWindowIcon(app_icon)
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    # Create handle prg for the Graphic Interface
    prg = FastScanMainWindow()

    # print('showing GUI')
    prg.show()

    try:
        app.exec_()
    except:
        print('exiting')


if __name__ == '__main__':

    main()
