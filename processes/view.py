# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.person import PersonMessage
from processes.pose import PoseMessage
from processes.camera import CameraMessage
from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import mediapipe as mp
import time

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

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            # resize image
            print("Person message received")
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Raw image", resized)
            cv2.waitKey(1)
        if type(msg) == PersonMessage:
            # resize image
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("ROI image", resized)
            cv2.waitKey(1)
        return None


def view(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("View started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = View(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending View")
    sleep(0.5)
    exit()