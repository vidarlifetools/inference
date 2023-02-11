# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.face import FaceMessage
from utilities.face_utils import face_prediction
import time


MODULE_FACECLASS = "Faceclass"

@dataclass
class FaceclassMessage:
    timestamp: float = 0.0
    valid: bool = True
    face_class: int = 0


@dataclass
class FaceclassConfig:
    name: str = ""
    model_filename_expression: str = ""


class Faceclass(DataModule):
    name = MODULE_FACECLASS
    config_class = FaceclassConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.face_prediction = face_prediction( self.config.model_filename_expression)

    def process_data_msg(self, msg):
        if type(msg) == FaceMessage:
            #self.logger.info(f"Faceclass processing started")
            face_class = 0
            if msg.valid:
                face_class = self.face_prediction.face_class(msg.landmarks)
            return FaceclassMessage(msg.timestamp, msg.valid, face_class)


def faceclass(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Faceclass(config, status_uri, data_in_uris, data_out_ur)
    print(f"Faceclass started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Faceclass")
    sleep(0.5)
    exit()