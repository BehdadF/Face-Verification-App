from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread
from utils.ImageUtils import ImageUtils
from utils.FaceModelUtils import FaceModelUtils

class FaceVerificationUI(QWidget):
    def __init__(self, verification_thread:QThread):
        super().__init__()
        self.verification_thread = verification_thread
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Source Directory Selection
        self.source_button = QPushButton("Select Source Directory")
        self.source_button.clicked.connect(self.select_source_directory)
        layout.addWidget(self.source_button)
        self.source_label = QLabel("No source selected")
        layout.addWidget(self.source_label)

        # Ref Image
        self.browse_button = QPushButton("Browse Image")
        self.capture_button = QPushButton("Capture Image")

        ## Image Preview
        self.image_label = QLabel("No Image Selected")
        self.image_label.setScaledContents(True)

        ## Add Widgets
        layout.addWidget(self.browse_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.image_label)

        ## Connect Buttons
        self.browse_button.clicked.connect(self.browse_image)
        self.capture_button.clicked.connect(self.capture_image)

        ## Last selected image (stored in memory)
        self.last_image = None  # Will hold a QImage

        # Target Directory Selection
        self.target_button = QPushButton("Select Target Directory")
        self.target_button.clicked.connect(self.select_target_directory)
        layout.addWidget(self.target_button)
        self.target_label = QLabel("No target selected")
        layout.addWidget(self.target_label)

        # Model Selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Facenet512", "ArcFace", "Facenet", "VGG-Face", "OpenFace"])
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)
        self.model_combo.currentTextChanged.connect(self.update_threshold)

        # Distance Selection
        self.distance_combo = QComboBox()
        self.distance_combo.addItems(["cosine", "euclidean", "euclidean_l2"])
        layout.addWidget(QLabel("Select Distance Metric:"))
        layout.addWidget(self.distance_combo)
        self.distance_combo.currentTextChanged.connect(self.update_threshold)

        # Threshold Input
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText(f"default: {FaceModelUtils.getThreshold(self.model_combo.currentText(), self.distance_combo.currentText())}")
        layout.addWidget(QLabel("Threshold:"))
        layout.addWidget(self.threshold_input)


        # Detection Model Selection
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["retinaface", "mtcnn", "opencv"])
        layout.addWidget(QLabel("Select Detection Model:"))
        layout.addWidget(self.detector_combo)

        # Move or Copy Option
        self.transfer_combo = QComboBox()
        self.transfer_combo.addItems(["Copy", "Move"])
        layout.addWidget(QLabel("Move or Copy:"))
        layout.addWidget(self.transfer_combo)

        # Progress Bar
        self.progress_label = QLabel("Idle")
        layout.addWidget(self.progress_label)

        # Run Button
        self.process_button = QPushButton("Run Face Verification")
        self.process_button.clicked.connect(self.run_verification)
        layout.addWidget(self.process_button)

        self.setLayout(layout)
        self.setWindowTitle("Face Verification App")
        self.show()

    def browse_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self,
                                                        "Select Image",
                                                        "",
                                                        "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img = ImageUtils.openImage(file_path)
            rgbImg = ImageUtils.bgrToRgb(img)
            image = self.numpy_to_qimage(rgbImg)

            if not image.isNull():
                self.input_img = img
                self.last_image = image
                self.update_preview(image)

    def capture_image(self):
        frame = ImageUtils.captureImg()
        if frame is not None:
            rgbFrame = ImageUtils.bgrToRgb(frame)

            qimage = self.numpy_to_qimage(rgbFrame)
            if qimage.isNull():
                print("Error: Captured image is null")
            else:
                self.input_img = frame
                self.last_image = qimage
                self.update_preview(qimage)
        else:
            print("Error: Failed to capture image")

    def numpy_to_qimage(self, np_img):
        h, w, ch = np_img.shape
        bytes_per_line = ch * w
        return QImage(np_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    
    def update_preview(self, image):
        self.image_label.setFixedSize(300, 300)
        pixmap = QPixmap.fromImage(image)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
        else:
            print("Error: Failed to set preview")

    def update_threshold(self):
        value = (FaceModelUtils.getThreshold(self.model_combo.currentText(), self.distance_combo.currentText()))
        self.threshold_input.setPlaceholderText(f"default: {value}")

    def select_source_directory(self):
        source_dir = QFileDialog.getExistingDirectory(self, "Select Source Directory")
        self.source_label.setText(f"Source Directory: {source_dir}")

    def select_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.input_label.setText(f"Input Image: {file_path}")

    def select_target_directory(self):
        target_dir = QFileDialog.getExistingDirectory(self, "Select Target Directory")
        self.target_label.setText(f"Target Directory: {target_dir}")

    def run_verification(self):
        if not self.source_label.text().startswith("Source Directory") or \
            not self.target_label.text().startswith("Target Directory") or \
            not self.last_image:
            print("Please fill out the required fields")
            print("source, reference image and target needs to be specified")
            return
        self.process_button.setEnabled(False)
        source_dir = self.source_label.text().split(": ")[1]
        target_dir = self.target_label.text().split(": ")[1]
        model = self.model_combo.currentText()
        threshold = float(self.threshold_input.text()) if self.threshold_input.text() else self.threshold_input.placeholderText().split(": ")[-1]
        distance_metric = self.distance_combo.currentText()
        transfer_method = self.transfer_combo.currentText()

        self.verification_thread.init(source_dir, target_dir, self.input_img, model, threshold, distance_metric, transfer_method)
        self.verification_thread.progress_signal.connect(self.progress_label.setText)
        self.verification_thread.finished_signal.connect(self.on_finished)
        self.verification_thread.start()

    def on_finished(self, elapsed_time):
        self.progress_label.setText(f"Verification Complete! Time: {elapsed_time:.2f} sec")
        self.process_button.setEnabled(True)