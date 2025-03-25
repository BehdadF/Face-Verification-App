from api.FileTransferer import FileTransferer
import shutil

class ShutilFileTransfer(FileTransferer):

    @staticmethod
    def copy(source, target):
        shutil.copy(source, target)

    @staticmethod
    def move(source,target):
        shutil.move(source, target)