# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep, time
import cv2

import json


MODULE_CAMERA = "Camera"

@dataclass
class CameraMessage:
    timestamp: float = 0.0
    fps: int = 15
    image: np.array = None


@dataclass
class CameraConfig:
    name: str = ""
    fps: int = 30
    video_path: str = ""


class Camera(DataModule):
    name = MODULE_CAMERA
    config_class = CameraConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.cap = cv2.VideoCapture(self.config.video_path)
        self.frame_no = 0

    def produce_data(self):

        sleep(1/self.config.fps)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_no += 1
                return CameraMessage(time(), self.config.fps, frame)
            self.cap.release()
        else:
            return None


def camera(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("Camera started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = Camera(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Camera")
    sleep(0.5)
    exit()