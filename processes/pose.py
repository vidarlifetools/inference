# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.person import PersonMessage
import cv2
import mediapipe as mp
import time
from feature_constants import keypoint_mapping_table
from gesture_utils import PoseFeature

MODULE_POSE = "Pose"

@dataclass
class PoseMessage:
    timestamp: float = 0.0
    valid: bool = True
    keypoints: np.array = None
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



class Face(DataModule):
    name = MODULE_POSE
    config_class = PoseConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.pose_feature = PoseFeature()
        pass
    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            #self.logger.info(f"Pose started processing at {time.time()}")
            pose_3d, valid, mp_pose_landmarks = self.pose_feature.get(msg.image)
            if valid:
                if self.config.view:
                    self.vew_pose(msg.image, mp_pose_landmarks)
                return PoseMessage(msg.timestamp, True, pose_3d, msg.image)
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
    proc = Face(config, status_uri, data_in_uris, data_out_ur)
    print(f"Pose started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Pose")
    sleep(0.5)
    exit()