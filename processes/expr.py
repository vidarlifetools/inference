# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.person import PersonMessage
import cv2
import mediapipe as mp
import time
from expr_utils import ExprFeature

MODULE_EXPR = "Expr"

@dataclass
class ExprMessage:
    timestamp: float = 0.0
    valid: bool = True
    landmarks: np.array = None
    image: np.array = None

@dataclass
class ExprConfig:
    name: str = ""
    view: bool = False


class Expr(DataModule):
    name = MODULE_EXPR
    config_class = ExprConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.expr_feature = ExprFeature()

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            face_landmarks, valid, mp_landmarks = self.expr_feature.get(msg.image)
            if valid:
                if self.config.view:
                    self.view_face(msg.image, mp_landmarks)
                return ExprMessage(msg.timestamp, True, face_landmarks, msg.image)
            else:
                return ExprMessage(msg.timestamp, False, None, msg.image)
        else:
            return None

    def view_face(self, image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        cv2.imshow('MediaPipe Expr Mesh', cv2.flip(image, 1))
        cv2.waitKey(1)


def expr(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Expr(config, status_uri, data_in_uris, data_out_ur)
    print(f"Expr started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Expr")
    sleep(0.5)
    exit()