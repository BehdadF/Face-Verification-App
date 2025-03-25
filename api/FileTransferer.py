from abc import ABC, abstractmethod

class FileTransferer(ABC):

    @abstractmethod
    def copy(source, target):
        pass

    @abstractmethod
    def move(source,target):
        pass