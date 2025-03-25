from deepface import DeepFace , modules

class FaceModelUtils:
    @staticmethod
    def getThreshold(model_name, distance_metric):
        return modules.verification.find_threshold(model_name, distance_metric)

    @staticmethod
    def extract_face(ref_img):
        """Load and extract face from reference image."""
        try:
            return DeepFace.extract_faces(ref_img, detector_backend='retinaface')[0]['face']
        except Exception as e:
            print(f"Error loading reference image: {e}")
            return None
