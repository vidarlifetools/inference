# import time
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.faceclass import FaceclassMessage
from processes.gestureclass import GestureclassMessage
from processes.soundclass import SoundclassMessage
from compound_utils import get_sequence_labels, get_sequence_labels_gesture, CompoundPrediction
import time
import numpy as np
from constants import *

MODULE_COMPOUND = "Compound"


@dataclass
class CompoundMessage:
    timestamp: float = 0.0
    compound_class: list = None
    frame_no: int = -1


@dataclass
class CompoundConfig:
    name: str = ""
    compound_buffer_len: int = 30
    nof_gesture_classes: int = 10
    nof_face_classes: int = 10
    nof_sound_classes: int = 10
    fps: int = 15
    model_filename_gesture: str = ""
    model_filename_face: str = ""
    model_filename_sound: str = ""
    model_filename_compound: str = ""


class Compound(DataModule):
    name = MODULE_COMPOUND
    config_class = CompoundConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.fps = self.config.fps
        self.compound_buffer_len = self.config.compound_buffer_len
        self.next_prediction_frame = self.config.compound_buffer_len  # TODO verify

        self.last_gesture_time = time.time()
        self.last_face_time = time.time()
        self.last_sound_time = time.time()
        self.sound_class = None
        self.gesture_class = [None, None, None, None]
        self.g_in = 0
        self.g_out = 0
        self.face_class = [None, None, None, None]
        self.e_in = 0
        self.e_out = 0

        self.buffer_length = len(self.face_class)

        self.face = []
        self.gesture = []
        self.sound = []
        self.face_frames = []
        self.gesture_sequence = []
        self.sound_frames = []
        self.frame_no = 0

        self.compound_prediction = CompoundPrediction(self.config.model_filename_compound,
                                                      self.config.model_filename_face,
                                                      self.config.model_filename_gesture,
                                                      self.config.model_filename_sound)


        # TODO Create self.compound_predict

    def process_data_msg(self, msg):
        if type(msg) == SoundclassMessage:
            self.logger.info(f"Sound ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f}, "
                             f"time since last = {time.time() - self.last_sound_time:.3f}")
            self.sound.append(msg.sound_class)
            self.sound_frames.append(msg.frame_no)
            #
            # self.last_sound_time = time.time()
            # self.sound_class = msg
        elif type(msg) == FaceclassMessage:
            self.logger.info(f"Expression ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f},"
                             f" time since last = {time.time() - self.last_face_time:.3f}")
            self.face.append(msg.face_class)
            self.face_frames.append(msg.frame_no)
            #
            # self.last_face_time = time.time()
            # self.expr_class[self.e_in] = msg
            # self.e_in = (self.e_in + 1) % self.buffer_length
        elif type(msg) == GestureclassMessage:
            self.logger.info(f"Gesture ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f},"
                             f" time since last = {time.time() - self.last_gesture_time:.3f}")
            self.gesture.append(msg.gesture_class)
            self.gesture_sequence.append([msg.frame_no-GESTURE_BUFFER_LEN, msg.frame_no])
            #
            # self.last_gesture_time = time.time()
            # self.gesture_class[self.g_in] = msg
            # self.g_in = (self.g_in + 1) % self.buffer_length
        else:
            pass


        # Find the current compound frame
        if np.all(np.array([len(self.sound_frames), len(self.gesture_sequence), len(self.face_frames)]) > 0):
            current_frame = np.min([self.sound_frames[-1], self.gesture_sequence[-1][1], self.face_frames[-1]])

            if current_frame == self.next_prediction_frame:
                first_frame = current_frame - self.compound_buffer_len
                last_frame = current_frame
                frame_sequence = (first_frame, last_frame)

                sequence_expr_classes, expr_frame_nos = get_sequence_labels(frame_sequence, self.face)
                sequence_sound_classes, sound_frame_nos = get_sequence_labels(frame_sequence, self.sound)
                sequence_gesture_classes, _ = get_sequence_labels(frame_sequence, self.gesture)
                gesture_sequence = self.gesture_sequence[first_frame:last_frame]

                compound_class = self.compound_prediction.get_class(sequence_gesture_classes, sequence_expr_classes,
                                                                    sequence_sound_classes,
                                                                    gesture_sequence, expr_frame_nos, sound_frame_nos,
                                                                    frame_sequence, current_frame, self.logger)
                # Update next prediction frame
                self.next_prediction_frame += 1
                print(msg)
                self.logger.info(f"Compound ({msg.timestamp}, frame_no {current_frame}): {compound_class}"
                                 f" processing time = {time.time() - msg.timestamp:.3f},"
                                 f" time since last = {time.time() - self.last_gesture_time:.3f}")
                return CompoundMessage(msg.timestamp, compound_class, current_frame)



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
