import sys
from os import path
import os
from PIL import Image
import json
import datetime
import speech_recognition as sr
from threading import Timer

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui




class Speech(QtCore.QThread):
    sig1 = QtCore.pyqtSignal(str)
    sig2 = QtCore.pyqtSignal(str)

    runing = True
    @QtCore.pyqtSlot()
    def run(self):
        print("worker")
        recogniser = sr.Recognizer()
        while self.runing:
            with sr.Microphone() as source:

                recogniser.adjust_for_ambient_noise(source)

                try:
                    self.sig2.emit("Please Speak")
                    audio = recogniser.listen(source, timeout=3)

                    print("done")
                except:
                    self.sig2.emit("Please Wait")
                    print("timeout")
                try:
                    text = recogniser.recognize_google(audio)
                    print(text)
                    if text != None:
                        if "yes" in text or "Yes" in text:
                            self.sig1.emit(text)
                        elif "no" in text or "No" in text:
                            self.sig1.emit(text)
                        # elif "in" in text or "in" in text:
                        #     self.sig1.emit(text)
                        # elif "out" in text or "out" in text:
                        #     self.sig1.emit(text)
                        # elif "break" in text or "break" in text:
                        #     self.sig1.emit(text)
                except:
                    print("error")

    def stop(self):
        self.runing = False

    def restart(self):
        self.runing = True
        # self.wait()

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
        self.path = "dataset" #path to saved images
        self.classifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.image = QtGui.QImage()
        self.database = {}

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
                #################################################################
                # Save the captured image into the datasets folder
                # print("fileName = ", "dataset/User." + str(self.num_of_ids) + '.' +
                #             str(self.count) + ".jpg")
                cv2.imwrite("dataset/User." + str(self.num_of_ids) + '.' +
                            str(self.count) + ".jpg", gray[y:y + h, x:x + w])
                #################################################################

        if (self.count == 70):
            self.train_classifier() #train the classfier with the saved images
            self.count += 1
            with open("face_id.json", "w") as outfile: #save the name and id in json file
                json.dump(self.name_to_id, outfile)
            self.database[self.face_id] = {'null' : {'in':'Null', 'out' : 'Null', 'break' : 'Null'} }
            self.face_name.emit("Registered")
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update() #keep running this class on the timer loop

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

class recognize_face(FaceDetectionWidget): #inheriting form the main class
    name = QtCore.pyqtSignal(str)
    signal = QtCore.pyqtSignal(str)

    def __init__(self, fp ):
        super().__init__(FaceDetectionWidget)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        print(str(len(self.name_to_id)) + '-------------------------------')
        if(len(self.name_to_id) > 0):
            self.recognizer.read('trainer/trainer.yml')
        else:
            self.signal.emit('k')
        # self.speech_thread = Speech()
        self.check_face = True
        self.show = True


    def recognize_face(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        minH = 64.0
        minW = 48.0
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        if len(faces) == 0:
            self.name.emit("No Face Detected")
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100 and confidence > 30):
                # print(id)
                id = self.name_to_id[str(id)]
                if self.show:
                    self.name.emit('Are you ' + str(id) + '?')
                    self.show = False
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                # if self.show:
                self.name.emit(id)

        self.image = self.get_qimage(image)
        #
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

class MainWindow(QtWidgets.QWidget):
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)

        fp = haarcascade_filepath

        self.thread = Speech()

        self.face_recognition_widget = recognize_face(haarcascade_filepath) #init face recognition class

        self.record_video = RecordVideo() #timer classf for looping

        if path.exists("face_id.json"):  # file that saves the face name to the id number
            f = open("face_id.json", )
            # returns JSON object as
            check = json.load(f)

        if path.exists("database.json"):  # file that saves the face name to the id number
            f = open("database.json", )

            # returns JSON object as
            self.database = json.load(f)

        else:
            self.database = {}



        #Adding Widget to the window
        self.check_lable = QtWidgets.QLabel() #first lable ro display name
        self.disp_speech = QtWidgets.QLabel()# label class to show options
        self.loading_lable  = QtWidgets.QLabel() #show registering loading
        self.check_lable.setAlignment(QtCore.Qt.AlignCenter)
        self.disp_speech.setAlignment(QtCore.Qt.AlignCenter)
        self.loading_lable.setAlignment(QtCore.Qt.AlignCenter)
        self.check_lable.setFont(QtGui.QFont('Arial', 15))
        self.disp_speech.setFont(QtGui.QFont('Arial', 15))
        self.loading_lable.setFont(QtGui.QFont('Arial', 15))
        self.loading_lable.setText("Enter Name of the Employee")
        self.name_of_user = QtWidgets.QLineEdit() #text box widget for taking name

        self.stacked1 = QtWidgets.QStackedLayout()
        self.stacked1.addWidget(self.check_lable)
        self.stacked1.addWidget(self.loading_lable)

        self.to_reg_win = QtWidgets.QPushButton('Register1')  # button
        self.register_btn = QtWidgets.QPushButton('Register2')  # button
        self.test_btn = QtWidgets.QPushButton('test')  # button

        self.stacked2 = QtWidgets.QStackedLayout()
        self.stacked2.addWidget(self.disp_speech)
        # self.stacked2.addWidget(self.test_btn)
        self.stacked2.addWidget(self.name_of_user)



        self.stacked3 = QtWidgets.QStackedLayout()
        self.stacked3.addWidget(self.to_reg_win)
        self.stacked3.addWidget(self.register_btn)


        self.check_lable.setText("Place Holder")

        layout = QtWidgets.QVBoxLayout()  # horizontal layout
        # adding the widgets to the window
        layout.addWidget(self.face_recognition_widget)
        layout.addLayout(self.stacked1)
        layout.addLayout(self.stacked2)
        layout.addLayout(self.stacked3)

        self.to_reg_win.hide() #show only when user wants to register

        self.record_video.image_data.connect(self.face_recognition_widget.recognize_face)  # emits frame from opencv to face detection class

        self.face_recognition_widget.name.connect(self.interact_with_text)

        self.to_reg_win.clicked.connect(self.admin_check) #opens admin passow dialog for passowrd

        self.register_btn.clicked.connect(self.send_data)

        self.face_recognition_widget.face_name.connect(self.check_registered)  # loading text

        self.thread.sig1.connect(self.check_name)

        self.thread.sig2.connect(self.disp_speech.setText)

        if (len(check) > 0):
            self.record_video.start_recording()  # start timer(for looping)
        else:
            self.admin_check()

        # self.face_recognition_widget.speech_thread.start()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.setLayout(layout)

    def interact_with_text(self, text):
        if self.record_video.timer.isActive() and "Are you" in text:
            self.check_lable.setText(text)
            # self.face_recognition_widget.show = False
            self.thread.start()
        elif self.record_video.timer.isActive() and "unknown" in text:
            self.admin_check()

    def send_data(self):
        self.face_recognition_widget.face_id = self.name_of_user.text()  # read the face id
        self.name_of_user.setEnabled(False)
        self.record_video.start_recording() #start timer(for looping)
        self.face_recognition_widget.count = 0 #count in the face detection class

    def check_name(self, text):
        self.face_recognition_widget.name.disconnect()
        self.thread.sig1.disconnect()
        self.thread.stop()
        self.employee_name = self.check_lable.text().replace('Are you ', '')
        self.employee_name = self.employee_name.replace('?', '')
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        if(text == "yes"):
            if(len(self.database) == 0):
                self.check_lable.setText("You are not clocked in. Do you want to Clock in?")
                self.thread.sig1.connect(self.clock_in)
                self.thread.restart()
            else:
                if(self.employee_name in self.database.keys()):
                    if(date in self.database[self.employee_name].keys()):
                        if(self.database[self.employee_name][date]['break']['out'] != ''):
                            self.check_lable.setText("Looks like you were on a break. Do you want to continue your shift?")
                            self.thread.sig1.connect(self.clock_break_in)
                            self.thread.restart()
                        else:
                            self.check_lable.setText("Looks like you are already clocked in today. Do you want to Clock out?")
                            self.thread.sig1.connect(self.clock_out)
                            self.thread.restart()
                    else:
                        self.database[self.employee_name] = {date : {"in": "", "out": "", "break": {"in": "", "out": ""}}}
                        self.check_lable.setText("You are not clocked in. Do you want to Clock in?")
                        self.thread.sig1.connect(self.clock_in)
                        self.thread.restart()
                else:
                    self.database[self.employee_name] = {date: {"in": "", "out": "", "break": {"in": "", "out": ""}}}
                    self.check_lable.setText("You are not clocked in. Do you want to Clock in?")
                    self.thread.sig1.connect(self.clock_in)
                    self.thread.restart()

            with open("database.json", "w") as outfile:  # save the name and id in json file
                json.dump(self.database, outfile)

        self.face_recognition_widget.name.connect(self.interact_with_text)

    def clock_in(self, text):
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        self.thread.stop()
        self.thread.sig1.disconnect()
        if("yes" in text):
            # self.database[self.employee_name] = {str(date): {'in': str(time), 'out': 'Null', 'break': {"in": "", "out": ""}}}
            self.database[self.employee_name][date]['in'] = time
            self.in_out_confirm('in')
            # cv2.imwrite(date+'_'+time+'_'+self.employee_name, self.face_recognition_widget.image)
        elif("no" in text):
            self.check_lable.setText("Do you want to FORCE Clock Out?")
            self.thread.sig1.connect(self.clock_out_execption)
            self.thread.restart()
        with open("database.json", "w") as outfile:  # save the name and id in json file
            json.dump(self.database, outfile)
        self.thread.restart()


    def clock_out(self, text):
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        print('out')
        self.thread.stop()
        self.thread.sig1.disconnect()
        if "yes" in text:
            self.database[self.employee_name][date]['out'] = time
            self.in_out_confirm('out')
            self.thread.sig1.connect(self.check_name)
        elif "no" in text:
            self.check_lable.setText("Do you want to go out on a break")
            self.thread.sig1.connect(self.clock_break)

        with open("database.json", "w") as outfile:  # save the name and id in json file
            json.dump(self.database, outfile)
        self.thread.restart()

    def clock_out_execption(self, text):
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        self.thread.stop()
        self.thread.sig1.disconnect()
        if text == "yes":
            self.database[self.employee_name][date]['in'] = "EXCEPTION"
            self.database[self.employee_name][date]['out'] = time
            self.in_out_confirm('out')
            self.thread.sig1.connect(self.check_name)
        if text == "no":
            self.thread.sig1.connect(self.check_name)
        with open("database.json", "w") as outfile:  # save the name and id in json file
            json.dump(self.database, outfile)
        self.thread.restart()

    def clock_break(self, text):
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        self.thread.sig1.disconnect()
        self.thread.stop()
        if text == "yes":
            self.database[self.employee_name][date]['break'] = {'in' : ''  , 'out' : time}

            with open("database.json", "w") as outfile:  # save the name and id in json file
                json.dump(self.database, outfile)
            self.in_out_confirm('break')
        self.thread.sig1.connect(self.check_name)
        self.thread.restart()

    def clock_break_in(self, text):
        date = datetime.datetime.now().strftime("%Y/%m/%d")
        time = datetime.datetime.now().strftime("%H:%M:%S")
        self.thread.sig1.disconnect()
        self.thread.stop()
        if"yes" in text:
            self.database[self.employee_name][date]['break'] = {'in': time, 'out': ''}
            with open("database.json", "w") as outfile:  # save the name and id in json file
                json.dump(self.database, outfile)
        self.thread.sig1.connect(self.check_name)
        self.thread.restart()

    def in_out_confirm(self, text):
        msgBox = QtWidgets.QMessageBox()
        print('-----' , text,  '-----')
        if('in' in text or 'out' in text):
            msgBox.setText("You Have been Clocked " + text)
        else:
            msgBox.setText("You are on a break")
        msgBox.setWindowTitle("Confiration box")
        self.face_recognition_widget.name.connect(self.interact_with_text)
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QtWidgets.QMessageBox.Ok:
            self.show()

    def check_registered(self, reg):
        self.loading_lable.setText(reg)
        if(reg == "Registered"):
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Are you done registering")
            msgBox.setWindowTitle("Confiration box")
            msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            returnValue = msgBox.exec()
            if returnValue == QtWidgets.QMessageBox.Ok:
                self.stacked1.setCurrentIndex(0)
                self.stacked2.setCurrentIndex(0)
                self.stacked3.setCurrentIndex(0)
                self.record_video.timer.stop()
                self.record_video.image_data.disconnect()
                self.record_video.image_data.connect(self.face_recognition_widget.recognize_face)
                self.record_video.start_recording()
                self.face_recognition_widget.recognizer.read('trainer/trainer.yml')
            elif returnValue == QtWidgets.QMessageBox.Cancel:
                self.name_of_user.setEnabled(True)

    def admin_check(self):
        msgBox = QtWidgets.QInputDialog()
        text, ok = QtWidgets.QInputDialog().getText(msgBox, "QInputDialog().getText()",
                                          "Please enter Admin Password", QtWidgets.QLineEdit.Normal)
        if ok:
          self.stacked1.setCurrentIndex(1)
          self.stacked2.setCurrentIndex(1)
          self.stacked3.setCurrentIndex(1)
          self.name_of_user.setEnabled(True)
          self.record_video.timer.stop()
          self.record_video.image_data.disconnect()
          self.record_video.image_data.connect(self.face_recognition_widget.image_data_slot)
        else: pass

if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir,
                                 'data',
                                 'haarcascade_frontalface_default.xml')
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    widget = MainWindow(cascade_filepath)
    win.setCentralWidget(widget)
    win.setMinimumSize(680, 700)
    win.show()


    sys.exit(app.exec_())