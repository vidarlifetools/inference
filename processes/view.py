# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.person import PersonMessage
from processes.pose import PoseMessage
from processes.face import FaceMessage
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
        self.mp_face_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        self.mp_pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.mp_drawing_style = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            # resize image
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('window-name', resized)
            cv2.waitKey(1)
        if type(msg) == FaceMessage:
            if msg.valid:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=msg.image,
                    landmark_list=msg.landmarks,
                    connections=self.mp_face_connections,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_style
                )
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(msg.image, 1))
            cv2.waitKey(1)
        if type(msg) == PoseMessage:
            mp.solutions.drawing_utils.draw_landmarks(
                msg.image,
                msg.landmarks,
                self.mp_face_connections,
                self.mp_drawing_style)
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(msg.image, 1))
            cv2.waitKey(1)

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
    print("Ending View")
    sleep(0.5)
    exit()