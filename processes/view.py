# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
from utilities.find_person import peoples
import cv2

import json


MODULE_VIEW = "View"

@dataclass
class ViewMessage:
    fps: int = 15
    image: np.array = None


@dataclass
class ViewConfig:
    name: str = ""
    scale: float = 1.0


class View(DataModule):
    name = MODULE_VIEW
    config_class = ViewConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.peoples = peoples()

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            boxes = self.peoples.detect(msg.image, threshold=0.9)
            print(f"Boxes = {boxes}")
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)

            # resize image
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('window-name', resized)
            cv2.waitKey(1)

            print(f"Grabber module: Process data msg")
        return None


def view(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("Camera started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = View(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending proc1")
    sleep(0.5)
    exit()