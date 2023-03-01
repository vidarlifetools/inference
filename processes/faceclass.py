# import time
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.face import FaceMessage
from face_utils import FacePrediction
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

        self.expr_prediction = FacePrediction( self.config.model_filename_expression)

    def process_data_msg(self, msg):
        if type(msg) == FaceMessage:
            #self.logger.info(f"Exprclass processing started")
            face_class = 0
            if msg.valid:
                face_class = self.expr_prediction.get_class(msg.landmarks)
            return FaceclassMessage(msg.timestamp, msg.valid, face_class)


def faceclass(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Faceclass(config, status_uri, data_in_uris, data_out_ur)
    print(f"Exprclass started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Exprclass")
    sleep(0.5)
    exit()