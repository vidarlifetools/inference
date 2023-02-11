# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import soundfile as sf

import time

import json


MODULE_AUDIO = "Audio"

@dataclass
class AudioMessage:
    timestamp: float = 0.0
    valid: bool = True
    sr: int = 16000
    audio: np.array = None


@dataclass
class AudioConfig:
    name: str = ""
    audio_path: str = ""


class Audio(DataModule):
    name = MODULE_AUDIO
    config_class = AudioConfig

    def __init__(self, *args):
        super().__init__(*args)
        sound_file = self.config.audio_path
        self.audio, self.sr = sf.read(sound_file)
        if len(self.audio.shape) > 1:
            self.audio = self.audio[:,0]
        self.audio_idx = 0

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            #self.logger.info(f"Sound processing started")
            # Create a buffer with audio samples covering time between frames
            no_of_samples = int(self.sr / msg.fps)
            if self.audio_idx + no_of_samples < len(self.audio):
                audio_buffer = self.audio[self.audio_idx:self.audio_idx+no_of_samples]
                self.audio_idx += no_of_samples
                return AudioMessage(msg.timestamp, True, self.sr, audio_buffer)
        #else:
        #    return AudioMessage(False, self.sr, None)


def audio(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Audio(config, status_uri, data_in_uris, data_out_ur)
    print(f"Audio started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Audio")
    sleep(0.5)
    exit()