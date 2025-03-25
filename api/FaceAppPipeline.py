from abc import ABC, abstractmethod

class FaceAppPipeline(ABC):
    @abstractmethod
    def process_images():
        pass

    @abstractmethod
    def shutdown():
        pass