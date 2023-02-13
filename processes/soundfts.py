# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.sound import AudioMessage
from utilities.sound_utils import sound_feature
from utilities.ring_buffer import ring_buffer

from utilities.person_utils import peoples
from utilities.draw import draw_bbox
import cv2
import soundfile as sf

import time

import json


MODULE_AUDIOFTS = "Audiofts"

@dataclass
class AudioftsMessage:
    timestamps: list = None
    valids: list = True
    features: list = None


@dataclass
class AudioftsConfig:
    name: str = ""
    mean_norm: bool = True
    windowing: bool = True
    pre_emphasis: bool = True
    ampl_norm: bool = False
    noise_reduce: bool = False
    n_mfcc: int = 12
    sample_rate: int = 16000
    window_length: float = 0.032
    window_step: float = 0.016
    feature_length: float = 0.512
    feature_step: float = 0.128
    use_pitch: bool = True


class Audiofts(DataModule):
    name = MODULE_AUDIOFTS
    config_class = AudioftsConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.ring_buffer = ring_buffer(self.config.sample_rate)
        self.audio_buffer_size = int(self.config.feature_length*self.config.sample_rate)
        self.audio_buffer_step = int(self.config.feature_step*self.config.sample_rate)
        self.audio_buffer = np.zeros((self.audio_buffer_size,), dtype=float)

        self.sound_feature = sound_feature(
            win_size=self.config.window_length,
            win_step=self.config.window_step,
            feat_size=self.config.feature_length,
            feat_step=self.config.feature_step,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            windowing=self.config.windowing,
            pre_emphasis=self.config.pre_emphasis,
            mean_normalization=self.config.mean_norm,
            noise_reduce=self.config.noise_reduce,
            ampl_normalization=self.config.ampl_norm,
            use_pitch=self.config.use_pitch

        )

    def process_data_msg(self, msg):
        if type(msg) == AudioMessage:
            #self.logger.info(f"Soundfts processing started")
            self.ring_buffer.put(msg.audio)
            timestamp = msg.timestamp
            timestamps = []
            valids = []
            features = []
            while self.ring_buffer.get_length() > self.audio_buffer_size:
                self.ring_buffer.get(self.audio_buffer, self.audio_buffer_size, self.audio_buffer_step)
                timestamps.append(timestamp)
                features.append(self.sound_feature.get_feature(self.audio_buffer))
                valids.append(True)
                timestamp += self.config.feature_step
            if len(features) > 0:
                return AudioftsMessage(timestamps, valids, features)


def audiofts(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Audiofts(config, status_uri, data_in_uris, data_out_ur)
    print(f"Audiofts started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Audiofts")
    sleep(0.5)
    exit()