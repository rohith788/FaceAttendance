import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui





if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,'dara', 'haarcascade_frontalface_default.xml')
    cascade_filepath = path.abspath(cascade_filepath)
    # main(cascade_filepath)
    print(cascade_filepath)