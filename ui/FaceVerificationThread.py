import time
from PyQt6.QtCore import QThread, pyqtSignal
from api.FaceAppPipeline import FaceAppPipeline

class FaceVerificationThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(float)

    def __init__(self, verifier:FaceAppPipeline):
        super().__init__()
        self.verifier = verifier

    def init(self, source, target, ref_img, model, threshold,
                            distance_metric, transfer_method):
        self.source = source
        self.target = target
        self.ref_img = ref_img
        self.model = model
        self.threshold = threshold
        self.distance_metric = distance_metric
        self.transfer_method = transfer_method

    def run(self):
        self.progress_signal.emit("Processing...")
        start = time.time()
        pipeline = self.verifier(self.source,
                                   self.target,
                                   self.ref_img,
                                   model_name=self.model,
                                   threshold=self.threshold,
                                   distance_metric=self.distance_metric,
                                   transfer_method=self.transfer_method)
        pipeline.process_images()
        pipeline.shutdown()
        end = time.time()
        self.progress_signal.emit("Processing complete!")
        elapsed_time = end - start
        self.finished_signal.emit(elapsed_time)