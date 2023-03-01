import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.faceclass import FaceclassMessage
from processes.gestureclass import GestureclassMessage
from processes.soundclass import SoundclassMessage
from compound_utils import get_sequence_labels, get_sequence_labels_gesture, CompoundPrediction
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
    model_filename_gesture: str = ""
    model_filename_face: str = ""
    model_filename_sound: str = ""
    model_filename_compound: str = ""


class Compound(DataModule):
    name = MODULE_COMPOUND
    config_class = CompoundConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.next_prediction_frame = COMPOUND_BUFFER_LEN

        # Initialize lists for face, gesture and sound
        self.face = []
        self.face_frames = []

        self.gesture = []
        self.gesture_sequence = []

        self.sound = []
        self.sound_frames = []

        # Initialize compound prediction model(s)
        self.compound_prediction = CompoundPrediction(self.config.model_filename_compound,
                                                      self.config.model_filename_face,
                                                      self.config.model_filename_gesture,
                                                      self.config.model_filename_sound)

    def process_data_msg(self, msg):
        if type(msg) == SoundclassMessage:
            self.logger.info(f"Sound ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f}")
            self.sound.append(msg.sound_class)
            self.sound_frames.append(msg.frame_no)
        elif type(msg) == FaceclassMessage:
            self.logger.info(f"Expression ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f}")
            self.face.append(msg.face_class)
            self.face_frames.append(msg.frame_no)
        elif type(msg) == GestureclassMessage:
            self.logger.info(f"Gesture ({msg.timestamp}, frame_no {msg.frame_no}):"
                             f" processing time = {time.time() - msg.timestamp:.3f}")
            self.gesture.append(msg.gesture_class)
            self.gesture_sequence.append([msg.frame_no - GESTURE_BUFFER_LEN, msg.frame_no])
        else:
            pass

        # Check that some frames have been added for all sub-models
        if np.all(np.array([len(self.sound_frames), len(self.gesture_sequence), len(self.face_frames)]) > 0):
            # Find latest frame number with predictions from all submodels
            current_frame = np.min([self.sound_frames[-1], self.gesture_sequence[-1][1], self.face_frames[-1]])

            # If current frame matches next prediction frame, run prediction
            if current_frame == self.next_prediction_frame:
                # Select compound window
                first_frame = current_frame - COMPOUND_BUFFER_LEN
                last_frame = current_frame
                frame_sequence = (first_frame, last_frame)

                # Retrieve classes and frame number for current compound window
                sequence_expr_classes, expr_frame_nos = get_sequence_labels(frame_sequence, self.face)
                sequence_sound_classes, sound_frame_nos = get_sequence_labels(frame_sequence, self.sound)
                sequence_gesture_classes, _ = get_sequence_labels(frame_sequence, self.gesture)
                gesture_sequence = self.gesture_sequence[first_frame:last_frame]

                # Get prediction
                compound_class = self.compound_prediction.get_class(sequence_gesture_classes, sequence_expr_classes,
                                                                    sequence_sound_classes,
                                                                    gesture_sequence, expr_frame_nos, sound_frame_nos,
                                                                    frame_sequence, current_frame, self.logger)
                # Update next prediction frame
                self.next_prediction_frame += 1

                # Log prediction
                self.logger.info(
                    f"Compound ({msg.timestamp}, frame_no {current_frame}): compound class {compound_class}"
                    f" processing time = {time.time() - msg.timestamp:.3f}")
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
