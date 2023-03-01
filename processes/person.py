# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
from person_utils import PersonBbox
import cv2
import time
import json

MODULE_PERSON = "Person"

@dataclass
class PersonMessage:
    timestamp: float = 0.0
    fps: int = 15
    image: np.array = None
    frame_no: int = -1


@dataclass
class PersonConfig:
    name: str = ""
    scale: float = 1.0
    tracking: bool = True
    tracking_threshold: float = 0.5
    tracking_info_path: str = ""


class Person(DataModule):
    name = MODULE_PERSON
    config_class = PersonConfig

    def __init__(self, *args):
        super().__init__(*args)
        if self.config.tracking:
            with open(self.config.tracking_info_path, "r") as fp:
                tracking_data = json.load(fp)
                if "target_person" in tracking_data.keys():
                    tracking_bbox = tracking_data["target_person"]
                    if "target_person_frame" in tracking_data.keys():
                        tracking_first_frame = tracking_data["target_person_frame"]
                    else:
                        tracking_first_frame = 0
                else:
                    self.config.tracking = False

        self.person = PersonBbox(self.config.tracking, tracking_bbox, tracking_first_frame, scale=0.5)
        print("Person initiated")

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            #self.logger.info(f"Person processing started")
            frame_no = msg.frame_no
            if self.config.scale != 1.0:
                width = int(msg.image.shape[1] * self.config.scale)
                height = int(msg.image.shape[0] * self.config.scale)
                dim = (width, height)
                resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)
            else:
                resized = msg.image

            bbox = self.person.detect(resized, threshold=0.9)
            if bbox is None:
                return None
            else:
                # Crop the image to the person
                x1, y1, x2, y2 = bbox
                return PersonMessage(msg.timestamp, msg.fps, resized[y1:y2, x1:x2].copy(),
                                     frame_no=frame_no)
        else:
            return None


def person(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Person(config, status_uri, data_in_uris, data_out_ur)
    print(f"Find_person started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Person")
    sleep(0.5)
    exit()