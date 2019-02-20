# -*- coding: utf-8 -*-
"""

@author: Steinn Ymir Agustsson

    Copyright (C) 2018 Steinn Ymir Agustsson, Vladimir Grigorev

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
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox

def raise_Qerror(doingWhat, errorHandle, type='Warning', popup=True):
    """ opens a dialog window showing the error"""
    errorMessage = 'Thread Error while {0}:\n{1}'.format(doingWhat, errorHandle)
    print(errorMessage)
    if popup:
        errorDialog = QMessageBox()
        errorDialog.setText(errorMessage)
        if type == 'Warning':
            errorDialog.setIcon(QMessageBox.Warning)
        elif type == 'Critical':
            errorDialog.setIcon(QMessageBox.Critical)
        errorDialog.setStandardButtons(QMessageBox.Ok)
        errorDialog.exec_()

def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print(exctype, value, traceback)
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

def recompile(folder):
    print('recompiling')
    uic.compileUiDir(folder, execute=True)
    print('done')

def main():
    pass



if __name__ == '__main__':
    main()
