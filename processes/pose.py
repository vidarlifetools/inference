# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.person import PersonMessage
import cv2
import mediapipe as mp


import time

import json


MODULE_POSE = "Pose"

@dataclass
class PoseMessage:
    valid: bool = True
    keypoints: np.array = None
    image: np.array = None


@dataclass
class PoseConfig:
    name: str = ""
    static_image_mode: bool = True
    model_complexity: int = 2
    enable_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

class Face(DataModule):
    name = MODULE_POSE
    config_class = PoseConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            enable_segmentation=self.config.enable_segmentation,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            results = self.mp_pose.process(msg.image)
            if results.pose_landmarks:
                return PoseMessage(True, results.pose_landmarks, msg.image)
            else:
                return PoseMessage(False, None, msg.image)
        else:
            return None


def pose(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("Pose started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = Face(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Pose")
    sleep(0.5)
    exit()