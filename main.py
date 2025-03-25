import sys
from PyQt6.QtWidgets import QApplication
from ui.FaceVerificationThread import FaceVerificationThread
from ui.FaceVerificationUI import FaceVerificationUI
from core.RayFacePipeline import RayFacePipeline


if __name__ == "__main__":
    face_verification_thread = FaceVerificationThread(RayFacePipeline)

    app = QApplication(sys.argv)
    window = FaceVerificationUI(face_verification_thread)
    sys.exit(app.exec())