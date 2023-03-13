# import time
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.pose import PoseMessage
from gesture_utils import GesturePrediction
import time
from constants import *


MODULE_GESTURECLASS = "Gestureclass"

@dataclass
class GestureclassMessage:
    timestamp: float = 0.0
    valid: bool = False
    gesture_class: int = 0
    frame_no: int = -1


@dataclass
class GestureclassConfig:
    name: str = ""
    model_filename_pose: str = ""
    model_filename_gesture: str = ""


class Gestureclass(DataModule):
    name = MODULE_GESTURECLASS
    config_class = GestureclassConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.gesture_prediction = GesturePrediction( self.config.model_filename_pose, self.config.model_filename_gesture, self.logger)

    def process_data_msg(self, msg):
        if type(msg) == PoseMessage:
            frame_no = msg.frame_no
            self.logger.debug(f"Gestureclass processing started")
            gesture_class = MISSING_CLASS_GESTURE
            if msg.valid:
                gesture_class, gesture_probs = self.gesture_prediction.get_class(msg.keypoints)
            return GestureclassMessage(msg.timestamp, msg.valid, gesture_class, frame_no=frame_no)


def gestureclass(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Gestureclass(config, status_uri, data_in_uris, data_out_ur)
    print(f"Gestureclass started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Gestureclass")
    sleep(0.5)
    exit()