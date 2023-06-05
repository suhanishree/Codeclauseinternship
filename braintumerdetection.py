import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Brain Tumor Detection")
        self.setGeometry(100, 100, 400, 300)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(10, 10, 380, 200)

        self.open_button = QPushButton("Open Image", self)
        self.open_button.setGeometry(10, 220, 180, 30)
        self.open_button.clicked.connect(self.open_image)

        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.setGeometry(200, 220, 180, 30)
        self.analyze_button.clicked.connect(self.analyze_image)

    def open_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_path = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")[0]

        if file_path:
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))

    def analyze_image(self):
        # Add your image analysis code here
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    return image

import tensorflow as tf

model = tf.keras.models.load_model('path_to_your_model')

def analyze_image(image_path):
    image = preprocess_image(image_path)
    image = tf.expand_dims(image, axis=0)
    prediction

