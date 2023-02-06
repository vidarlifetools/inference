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
    timestamp: float = 0.0
    valid: bool = True
    landmarks: np.array = None
    image: np.array = None


@dataclass
class PoseConfig:
    name: str = ""
    view:bool = False
    static_image_mode: bool = True
    model_complexity: int = 2
    enable_segmentation: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

# Map from Mediapipe pose to old pose model with 17 keypoints
mapping_table = [0, -1, 1, -1, -1, 2, -1, 3, 4, -1, -1, 5,6,7,8,9,10,-1,-1,-1,-1,-1,-1,11 ,12, 13, 14, 15, 16, -1, -1, -1, -1]


class Face(DataModule):
    name = MODULE_POSE
    config_class = PoseConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.mp_pose = mp.solutions.pose.Pose(
            #static_image_mode=self.config.static_image_mode,
            #model_complexity=self.config.model_complexity,
            #enable_segmentation=self.config.enable_segmentation,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            results = self.mp_pose.process(msg.image)
            if results.pose_world_landmarks.landmark:
                if self.config.view:
                    self.vew_pose(msg.image, results.pose_landmarks)

                pose_3d = np.zeros((17, 5), dtype=np.float32)
                for i, landmark in enumerate(results.pose_world_landmarks.landmark):
                    if mapping_table[i] != -1:
                        conf = landmark.visibility if (mapping_table[i] > 12 and landmark.visibility > 0.8) or (
                                    mapping_table[i] <= 12) else 0.0
                        pose_3d[mapping_table[i], :] = [landmark.x * 1000.0, landmark.y * 1000.0, landmark.z * 1000.0,
                                                        conf, 0.0]
            else:
                return PoseMessage(msg.timestamp, False, None, msg.image)
        else:
            return None
    def vew_pose(self, image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        cv2.waitKey(1)


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