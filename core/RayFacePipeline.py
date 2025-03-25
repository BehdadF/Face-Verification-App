import os
import ray
from ray.util.queue import Queue
from core.FaceExtractor import FaceExtractor
from core.FaceVerifier import FaceVerifier
from utils import FileTransfererUtils
from utils.FaceModelUtils import FaceModelUtils
from utils.FileTransfererUtils import ShutilFileTransfer
from utils.SystemUtils import SystemUtils
from api.FaceAppPipeline import FaceAppPipeline

class RayFacePipeline(FaceAppPipeline):
    def __init__(self,
                    input_dir,
                    output_dir,
                    ref_img_path,
                    model_name="Facenet512",
                    threshold=0.3,
                    distance_metric='cosine',
                    transfer_method = 'copy',
                    transferer:FileTransfererUtils = ShutilFileTransfer,
                    num_extractors=int(0.75 * (SystemUtils.get_num_cpus() - 5)),
                    num_verifiers=int(0.25 * (SystemUtils.get_num_cpus() - 5))):
        ray.init(ignore_reinit_error=True)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ref_img_path = ref_img_path
        self.model_name = model_name
        self.threshold = threshold
        self.distance_metric = distance_metric
        self.transfer_method = transfer_method
        self.transferer = transferer
        self.queue = Queue()

        self.extractors = [FaceExtractor.remote(self.queue) for _ in range(num_extractors)]
        self.ref_img = self.load_reference_image()
        self.verifiers = [FaceVerifier.remote(self.ref_img,
                                              output_dir,
                                              self.queue,
                                              self.model_name,
                                              self.threshold,
                                              self.distance_metric,
                                              self.transfer_method,
                                              self.transferer) for _ in range(num_verifiers)]

    def load_reference_image(self):
        """Load and extract face from reference image."""
        return FaceModelUtils.extract_face(self.ref_img_path)

    def process_images(self):
        """Distribute tasks among extractors and verifiers."""
        img_paths = [
            os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) 
            if f.lower().endswith((".jpg", ".png"))
        ]

        verification_tasks = [verifier.verify_and_transfer.remote() for verifier in self.verifiers]

        extraction_tasks = [self.extractors[i % len(self.extractors)].extract_faces.remote(img) for i, img in enumerate(img_paths)]
        
        ray.get(extraction_tasks)

        for _ in range(len(self.verifiers)):
            self.queue.put(None)  # Signals workers to stop

        ray.get(verification_tasks)

        print("(Pipeline) Processing complete.")

    def shutdown(self):
        """Shutdown Ray cluster."""
        ray.shutdown()

