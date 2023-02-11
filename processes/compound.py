# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.faceclass import FaceclassMessage
from processes.soundclass import SoundclassMessage
from processes.pose import PoseMessage
from processes.camera import CameraMessage
from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import mediapipe as mp
import time

import json


MODULE_COMPOUND = "Compound"

@dataclass
class CompoundMessage:
    timestamp: float = 0.0
    compound_class: list = None

@dataclass
class CompoundConfig:
    name: str = ""


class Compound(DataModule):
    name = MODULE_COMPOUND
    config_class = CompoundConfig

    def __init__(self, *args):
        super().__init__(*args)

    def process_data_msg(self, msg):
        if type(msg) == FaceclassMessage:
            self.logger.info(f"Face class with timetag {msg.timestamp} received")
        if type(msg) == SoundclassMessage:
            for i in range(len(msg.timestamp)):
                self.logger.info(f"Sound class [{msg.sound_class[i]}]with timetag {msg.timestamp[i]} received")


def compound(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Compound(config, status_uri, data_in_uris, data_out_ur)
    print(f"Compound started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Compound")
    sleep(0.5)
    exit()