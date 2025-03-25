import ray
from deepface import DeepFace
import os
from queue import Empty
import time
from utils.FileTransfererUtils import ShutilFileTransfer, FileTransferer

@ray.remote
class FaceVerifier:
    def __init__(self,
                    ref_img,
                    output_dir,
                    queue,
                    model_name="Facenet512",
                    threshold=0.3,
                    distace_metric='cosine',
                    transfer_method = 'copy',
                    transferer:FileTransferer = ShutilFileTransfer):
        self.ref_img = ref_img
        self.output_dir = output_dir
        self.queue = queue
        self.model_name = model_name
        self.threshold = float(threshold)
        self.distance_metric = distace_metric
        self.transfer_method = transfer_method
        self.transferer = transferer

    def verify_and_transfer(self):
        while True:
            try:
                data = self.queue.get(timeout=10)

                if data is None:
                    print("breaking")
                    break
                img_path, face = data

                result = DeepFace.verify(face * 255, self.ref_img * 255, model_name=self.model_name, detector_backend="skip", distance_metric=self.distance_metric)

                if result['distance'] < self.threshold:
                    print(f"(FaceVerifier) Verified and copied {img_path} to {os.path.join(self.output_dir, os.path.basename(img_path))}")
                    self.transfer(self.transfer_method, img_path, os.path.join(self.output_dir, os.path.basename(img_path)))

                else:
                    print(f"(FaceVerifier) face not verified {img_path} - distance: {result['distance']}")

            except Empty:
                print("Queue is empty")
                time.sleep(2)

    def transfer(self, type:str, source, target):
        if type.lower() == 'move':
            self.transferer.move(source, target)
        elif type.lower() == 'copy':
            self.transferer.copy(source, target)