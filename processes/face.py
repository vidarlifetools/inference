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


MODULE_FACE = "Face"

@dataclass
class FaceMessage:
    timestamp: float = 0.0
    valid: bool = True
    landmarks: np.array = None
    image: np.array = None


@dataclass
class FaceConfig:
    name: str = ""
    view: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class Face(DataModule):
    name = MODULE_FACE
    config_class = FaceConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.mediapipe_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

    def process_data_msg(self, msg):
        if type(msg) == PersonMessage:
            #self.logger.info(f"Face processing started")
            results = self.mediapipe_face.process(msg.image)
            face_landmarks = np.zeros((468, 3), dtype=float)
            if results.multi_face_landmarks:
                if self.config.view:
                    self.view_face(msg.image, results.multi_face_landmarks[0])
                first = True
                for landmarks in results.multi_face_landmarks:
                    i=0
                    if first:
                        for landmark in landmarks.landmark:
                            face_landmarks[i, :] = [landmark.x * msg.image.shape[1],
                                                    landmark.y * msg.image.shape[0],
                                                    landmark.z * -1000.0]
                            i += 1
                face_landmarks = np.matmul(face_landmarks, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                #print(f"Face landmarks found, {len(results.multi_face_landmarks)}, {len(results.multi_face_landmarks[0])}")
                return FaceMessage(msg.timestamp, True, face_landmarks, msg.image)
            else:
                return FaceMessage(msg.timestamp, False, None, msg.image)
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
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        cv2.waitKey(1)


def face(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Face(config, status_uri, data_in_uris, data_out_ur)
    print(f"Face started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Face")
    sleep(0.5)
    exit()