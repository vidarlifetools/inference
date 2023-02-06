# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.audio import AudioMessage
from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import soundfile as sf

import time

import json


MODULE_AUDIOFTS = "Audiofts"

@dataclass
class AudioftsMessage:
    timestamp: float = 0.0
    valid: bool = True
    feature: np.array = None


@dataclass
class AudioftsConfig:
    name: str = ""
    mean_norm: bool = True
    windowing: bool = True
    pre_emphasis: bool = True
    n_mfcc: int = 12
    sample_rate: int = 16000
    window_length: float = 0.032
    window_step: float = 0.016
    feature_length: float = 0.512
    feature_step: float = 0.128
    use_pitch: bool = True

class ring_buffer:
    def __init__(self, size):
        self.buffer = np.zeros((size,), dtype=float)
        self.in_ptr = 0
        self.out_ptr = 0
        self.size = size

    def put(self, data):
        for i in range(data.shape[0]):
            self.buffer[self.in_ptr] = data[i]
            self.in_ptr = (self.in_ptr + 1)%self.size

    def get(self, data, length, step):
        ptr = self.out_ptr
        for i in range(length):
            data[i] = self.buffer[ptr]
            ptr = (ptr + 1)%self.size
        self.out_ptr = (self.out_ptr + step)%self.size

    def get_length(self):
        if self.in_ptr - self.out_ptr >= 0:
            length = self.in_ptr - self.out_ptr
        else:
            length = self.size + self.in_ptr - self.out_ptr
        return length




class Audiofts(DataModule):
    name = MODULE_AUDIOFTS
    config_class = AudioftsConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.ring_buffer = ring_buffer(self.config.sample_rate)
        self.audio_buffer_size = int(self.config.window_length*self.config.sample_rate)
        self.audio_buffer_step = int(self.config.window_step*self.config.sample_rate)
        self.audio_buffer = np.zeros((self.audio_buffer_size,), dtype=float)

    def process_data_msg(self, msg):
        if type(msg) == AudioMessage:
            self.ring_buffer.put(msg.audio)
            while self.ring_buffer.get_length() > self.audio_buffer_size:
                self.ring_buffer.get(self.audio_buffer, self.audio_buffer_size, self.audio_buffer_step)

            return AudioftsMessage(msg.timestamp, False, None)


def audiofts(start, stop, config, status_uri, data_in_uris, data_out_ur):
    print("Audiofts started", status_uri, data_in_uris, data_out_ur, flush=True)
    proc = Audiofts(config, status_uri, data_in_uris, data_out_ur)
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Audiofts")
    sleep(0.5)
    exit()