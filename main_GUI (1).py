import sys
import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QSlider, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from ASL import mediapipe_detection, draw_landmark, extract_keypoints, model, actions
import mediapipe as mp

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Subtitles")
        self.setGeometry(100, 100, 1280, 720)
        self.dark_mode = True

        self.camera_index = 1
        self.capture = None
        self.timer = QTimer()
        self.subtitle_size = 24
        self.virtual_camera = None

        self.sequence = []
        self.sentence = []
        self.predictions=[]
        self.threshold = 0.85
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['categorical_accuracy'])
        model.load_weights("model2.keras")

        self.init_ui()
        self.update_stylesheet()

    def init_ui(self):
        self.image_label = QLabel("Camera Feed")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000;")

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.theme_toggle = QCheckBox("Dark Mode")
        self.theme_toggle.setChecked(True)
        self.subtitle_slider = QSlider(Qt.Orientation.Horizontal)

        self.start_button.clicked.connect(self.start_feed)
        self.stop_button.clicked.connect(self.stop_feed)
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        self.subtitle_slider.valueChanged.connect(self.change_subtitle_size)

        self.subtitle_slider.setMinimum(12)
        self.subtitle_slider.setMaximum(48)
        self.subtitle_slider.setValue(self.subtitle_size)

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(QLabel("Subtitle Size:"))
        control_layout.addWidget(self.subtitle_slider)
        control_layout.addWidget(self.theme_toggle)
        control_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 4)
        main_layout.addLayout(control_layout, 1)

        self.setLayout(main_layout)

    def start_feed(self):
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            print("Error: Could not open webcam.")
            return

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.capture.get(cv2.CAP_PROP_FPS)) or 60

        # self.virtual_camera = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR)

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // fps)

    def stop_feed(self):
        self.sentence=[]
        self.sequence=[]
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.capture = None
        self.virtual_camera = None
        self.image_label.clear()

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        image, results = mediapipe_detection(frame, self.holistic)
        draw_landmark(image, results)

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-60:]

        if len(self.sequence) == 60:
            res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
            self.predictions.append(np.argmax(res))
            if np.unique(self.predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > self.threshold:
                    if len(self.sentence) == 0 or actions[np.argmax(res)] != self.sentence[-1]:
                        self.sentence.append(actions[np.argmax(res)])
        if len(self.sentence) > 5:
            self.sentence = self.sentence[-5:]

        text = " ".join(self.sentence)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
        center_x = image.shape[1] // 2
        y_offset = image.shape[0] - 20

        rect_width = text_width + 40
        rect_height = text_height + 20
        top_left = (center_x - rect_width // 2, y_offset - rect_height)
        bottom_right = (center_x + rect_width // 2, y_offset)

        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        cv2.putText(image, text, (center_x - text_width // 2, y_offset - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Virtual camera output (full resolution)
        # if self.virtual_camera:
        #     self.virtual_camera.send(image)
        #     self.virtual_camera.sleep_until_next_frame()

        # Resize for GUI preview
        resized_image = cv2.resize(image, (1280, 720))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0],
                          QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def change_subtitle_size(self, value):
        self.subtitle_size = value

    def toggle_theme(self):
        self.dark_mode = self.theme_toggle.isChecked()
        self.update_stylesheet()

    def update_stylesheet(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QWidget { background-color: #1e1e2f; color: white; font-family: 'Arial'; }
                QPushButton { background-color: #3a3f5c; border-radius: 10px; padding: 10px; }
                QPushButton:hover { background-color: #5c6699; }
                QSlider::handle:horizontal { background: white; border-radius: 5px; }
            """)
        else:
            self.setStyleSheet("""
                QWidget { background-color: #f0f0f0; color: #111; font-family: 'Arial'; }
                QPushButton { background-color: #d0d0e1; border-radius: 10px; padding: 10px; }
                QPushButton:hover { background-color: #b0b0d1; }
                QSlider::handle:horizontal { background: #111; border-radius: 5px; }
            """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())