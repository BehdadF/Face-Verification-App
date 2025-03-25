import ray
from deepface import DeepFace

@ray.remote
class FaceExtractor:
    """Ray actor for face extraction."""

    def __init__(self, queue):
        self.queue = queue  

    def extract_faces(self, img_path):
        """Extract faces and push them to the queue."""
        try:
            faces = DeepFace.extract_faces(img_path, detector_backend='retinaface')
            for face in faces:
                self.queue.put((img_path, face['face']))
            print(f"(FaceExtractor) Added faces from {img_path} to queue")
        except Exception as e:
            print(f"(FaceExtractor) Error extracting faces from {img_path}: No faces found!")
        return True  # Notify that this image is processed