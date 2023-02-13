# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.exprclass import ExprclassMessage
from processes.gestureclass import GestureclassMessage
from processes.soundclass import SoundclassMessage
from processes.pose import PoseMessage
from processes.camera import CameraMessage
from utilities.person_utils import peoples
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
    nof_gesture_classes: int = 10
    nof_expr_classes: int = 10
    nof_sound_classes: int = 10


class Compound(DataModule):
    name = MODULE_COMPOUND
    config_class = CompoundConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.last_gesture_time = time.time()
        self.last_face_time = time.time()
        self.last_sound_time = time.time()

    def process_data_msg(self, msg):
        if type(msg) == ExprclassMessage:
            #self.logger.info(
            #    f"Expression ({msg.timestamp}): processing time = {time.time() - msg.timestamp:.3f}, time since last = {time.time() - self.last_face_time:.3f}")
            self.last_face_time = time.time()
        if type(msg) == SoundclassMessage:
            self.logger.info(
                f"Sound ({msg.timestamp}): processing time = {time.time() - msg.timestamp[0]:.3f}, time since last = {time.time() - self.last_sound_time:.3f}")
            self.last_sound_time = time.time()
        if type(msg) == GestureclassMessage:
            #self.logger.info(f"Gesture ({msg.timestamp}): processing time = {time.time()-msg.timestamp:.3f}, time since last = {time.time()-self.last_gesture_time:.3f}")
            self.last_gesture_time = time.time()


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