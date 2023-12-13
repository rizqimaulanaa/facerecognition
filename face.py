import sys
import cv2
import face_recognition
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.video_capture = cv2.VideoCapture(0)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.video_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # Update frame setiap 50 ms

        # Load gambar contoh wajah dan nama
        self.known_face_images = []
        self.known_face_encodings = []
        self.known_face_names = []

        # Tambahkan gambar wajah dan nama untuk setiap orang
        self.add_known_face("C:\\Users\\PATH\\Pictures\\unknown1.jpg", "Nama")
        self.add_known_face("C:\\Users\\PATH\\Pictures\\unknown2.jpg", "Nama")
        self.add_known_face("C:\\Users\\PATH\\Pictures\\unknown3.jpg", "Nama")
        # Tambahkan lebih banyak gambar dan nama di sini dengan memanggil add_known_face lagi

    def add_known_face(self, image_path, name):
        face_image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]

        self.known_face_images.append(face_image)
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

    def update_frame(self):
        ret, frame = self.video_capture.read()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                for i, match in enumerate(matches):
                    if match:
                        name = self.known_face_names[i]
                        break

                # Gambar kotak di sekitar wajah
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Mengatur warna teks nama menjadi hijau
                color = (0, 255, 0)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, color, 1)

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Menggambar background gambar dengan warna putih untuk meningkatkan kontras
            q_image_with_bg = QImage(w, h, QImage.Format_RGB888)
            q_image_with_bg.fill(QColor(255, 255, 255))

            # Menggunakan QPainter untuk menggambar
            painter = QPainter(q_image_with_bg)
            painter.drawImage(0, 0, q_image)
            painter.end()  # Penting untuk mengakhiri QPainter

            self.video_label.setPixmap(QPixmap.fromImage(q_image_with_bg))

    def closeEvent(self, event):
        self.video_capture.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.setWindowTitle('FACE RECOGNITION')
    window.setGeometry(100, 100, 640, 480)
    window.show()
    sys.exit(app.exec_())
