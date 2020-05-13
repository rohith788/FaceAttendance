import sys
from os import path
import os
from PIL import Image
import json

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
        self.camera.set(3, 1080)  # set Width
        self.camera.set(4, 720)  # set Height]
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
        # print('fsddsdfsdfsdfsdfsdfsdfsdfsdfsdfsdfdfsdfsdfsfsdfsdsf')
        # print(haar_cascade_filepath)
        self.path = "dataset" #path to saved images
        self.classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.image = QtGui.QImage()

        self._width = 2
        self._min_size = (30, 30)
        self.count = 0
        self.face_id = 0
        self.c = 1

        if path.exists("face_id.json"): #file that saves the face name to the id number
            f = open("face_id.json", )

            # returns JSON object as
            self.name_to_id = json.load(f)

        else: #if file does not exist
            self.name_to_id = {}


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
        # if(self.face_id in self.name_to_id): self.face_name.emit("User Already exists. Please enter a different Name")
        faces = self.detect_faces(image_data)
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        if(self.count < 70):
            for (x, y, w, h) in faces:
                if(self.c == 7): self.c = 1
                txt = "Registering"
                for i in range(self.c):
                    txt += "."
                self.c += 1
                self.face_name.emit(txt)
                cv2.rectangle(image_data, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.count += 1
                if self.face_id not in self.name_to_id.values():
                    self.num_of_ids = len(self.name_to_id)
                    self.name_to_id[str(self.num_of_ids + 1)] = self.face_id
                if self.face_id in self.name_to_id.values():
                    self.num_of_ids = int(list(self.name_to_id.keys())[list(self.name_to_id.values()).index(self.face_id)])
                # print('index', self.num_of_ids)
                #################################################################
                # Save the captured image into the datasets folder
                # print("fileName = ", "dataset/User." + str(self.num_of_ids) + '.' +
                #             str(self.count) + ".jpg")
                cv2.imwrite("dataset/User." + str(self.num_of_ids) + '.' +
                            str(self.count) + ".jpg", gray[y:y + h, x:x + w])
                #################################################################

        if (self.count == 70):
            self.train_classifier()
            self.count += 1
            with open("face_id.json", "w") as outfile:
                json.dump(self.name_to_id, outfile)
            self.face_name.emit("Registered")
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def train_classifier(self):
        faces, ids = self.data_for_trainer()
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write('trainer/trainer.yml')

    def data_for_trainer(self):
        imagePaths = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.classifier.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

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
        self.face_detection_widget = FaceDetectionWidget(fp) #initializing the face detection class

        self.record_video = RecordVideo() #looping class for opencv

        self.id = QtWidgets.QLineEdit() #text box widget

        image_data_slot = self.face_detection_widget.image_data_slot #function for face detection
        self.record_video.image_data.connect(image_data_slot) #emits frame from opencv to face detection class
        # self.record_video.start_recording()
        lable = QtWidgets.QLabel() #label class
        self.run_button = QtWidgets.QPushButton('Start') #button
        lable.setText("Place Holder")

        layout = QtWidgets.QVBoxLayout() #horizontal layout
        #adding the widgets to the window
        layout.addWidget(self.face_detection_widget)
        layout.addWidget(self.run_button)
        layout.addWidget(lable)
        layout.addWidget(self.id)

        self.face_detection_widget.face_name.connect(lable.setText) #text in the label
        self.run_button.clicked.connect(self.send_data) #button click function
        self.setLayout(layout)

    def send_data(self):
        self.face_detection_widget.face_id = self.id.text()  # read the face id
        self.record_video.start_recording() #start timer(for looping)
        self.face_detection_widget.count = 0 #count in the face detection class




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