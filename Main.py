import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(3, 640)  # set Width
        self.camera.set(4, 480)  # set Height]
        self.timer = QtCore.QBasicTimer()


    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()

        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    face_name = QtCore.pyqtSignal(str)
    def __init__(self, haar_cascade_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.count = 0
        self.face_id = "default"

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)

        return faces

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        self.face_name.emit("no-name")
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        if(self.count < 50):
            for (x, y, w, h) in faces:
                self.face_name.emit("name")
                cv2.rectangle(image_data, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(self.face_id) + '.' +
                            str(self.count) + ".jpg", gray[y:y + h, x:x + w])
            # cv2.rectangle(image_data,
            #               (x, y),
            #               (x+w, y+h),
            #               self._red,
            #               self._width)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp)

        # TODO: set video port
        self.record_video = RecordVideo()

        self.id = QtWidgets.QLineEdit()



        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)
        # self.record_video.start_recording()
        lable = QtWidgets.QLabel()
        lable.setText("Place Holder")



        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)
        layout.addWidget(lable)
        layout.addWidget(self.id)
        # self.record_video.start_recording()
        self.face_detection_widget.face_name.connect(lable.setText)
        self.run_button.clicked.connect(self.send_data)
        self.setLayout(layout)
    def send_data(self):
        self.record_video.start_recording()
        self.face_detection_widget.face_id = self.id.text()


def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 'data',
                                 'haarcascade_frontalface_default.xml')

    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)