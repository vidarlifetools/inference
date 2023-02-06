# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import time

import json


MODULE_PERSON = "Person"

@dataclass
class PersonMessage:
    fps: int = 15
    image: np.array = None


@dataclass
class PersonConfig:
    name: str = ""
    scale: float = 1.0


class Person(DataModule):
    name = MODULE_PERSON
    config_class = PersonConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.peoples = peoples()
        print("Person initiated")

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)

            # resize image
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            boxes = self.peoples.detect(resized, threshold=0.9)
            if boxes[0] is None:
                return None
            else:
                # Crop the image to the person
                x1, y1, x2, y2 = boxes[0]
                return PersonMessage(msg.fps, resized[y1:y2, x1:x2].copy() )
        else:
            return None


def person(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("Find_person started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = Person(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Person")
    sleep(0.5)
    exit()