# import time
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.exprclass import ExprclassMessage
from processes.gestureclass import GestureclassMessage
from processes.soundclass import SoundclassMessage
import time

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
        self.sound_class = None
        self.gesture_class = [None, None, None, None]
        self.g_in = 0
        self.g_out = 0
        self.expr_class = [None, None, None, None]
        self.e_in = 0
        self.e_out = 0
        self.buffer_length = len(self.expr_class)
        # TODO Create self.compound_predict

    def process_data_msg(self, msg):
        if type(msg) == SoundclassMessage:
            self.logger.info(
                f"Sound ({msg.timestamp}): processing time = {time.time() - msg.timestamp:.3f}, time since last = {time.time() - self.last_sound_time:.3f}")
            self.last_sound_time = time.time()
            self.sound_class = msg
        if type(msg) == ExprclassMessage:
            #self.logger.info(
            #    f"Expression ({msg.timestamp}): processing time = {time.time() - msg.timestamp:.3f}, time since last = {time.time() - self.last_face_time:.3f}")
            self.last_face_time = time.time()
            self.expr_class[self.e_in] = msg
            self.e_in = (self.e_in + 1)%self.buffer_length
        if type(msg) == GestureclassMessage:
            #self.logger.info(f"Gesture ({msg.timestamp}): processing time = {time.time()-msg.timestamp:.3f}, time since last = {time.time()-self.last_gesture_time:.3f}")
            self.last_gesture_time = time.time()
            self.gesture_class[self.g_in] = msg
            self.g_in = (self.g_in + 1)%self.buffer_length

        if self.g_in != self.g_out and self.e_in != self.e_out:
            # There are expression and gesture classes present
            if abs(self.expr_class[self.e_out].timestamp - self.gesture_class[self.g_out].timestamp) < 0.01:
                # TODO Call self.compound_predict(self.gesture_class[], self.expr_class[], self.sound_class)
                gesture_class = self.gesture_class[self.g_out].gesture_class
                timestamp = self.gesture_class[self.g_out].timestamp
                self.g_out = (self.g_out + 1) % self.buffer_length
                self.e_out = (self.e_out + 1) % self.buffer_length
                return (CompoundMessage(timestamp, gesture_class))
            else:
                self.logger.error(f"Timestamps does not match, {self.expr_class}, {self.gesture_class}")



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