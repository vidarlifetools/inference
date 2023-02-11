# import time
import threading
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.soundfts import AudioftsMessage
from utilities.sound_utils import sound_prediction
import pickle
from utilities.sound_utils import sound_feature
from utilities.ring_buffer import ring_buffer

from utilities.find_person import peoples
from utilities.draw import draw_bbox
import cv2
import soundfile as sf

import time

import json


MODULE_SOUNDCLASS = "Soundclass"

@dataclass
class SoundclassMessage:
    timestamp: list = None
    valid: list = None
    sound_class: list = None


@dataclass
class SoundclassConfig:
    name: str = ""
    histogram_depth: int =  6
    histogram_limit: float = 0.5
    model_filename_client: str = ""
    model_filename_sound: str = ""
    model_filename_split_sound: str = ""


class Soundclass(DataModule):
    name = MODULE_SOUNDCLASS
    config_class = SoundclassConfig

    def __init__(self, *args):
        super().__init__(*args)

        self.sound_prediction = sound_prediction(
            self.config.model_filename_split_sound,
            self.config.model_filename_client,
            self.config.model_filename_sound,
            self.config.histogram_depth,
            self.config.histogram_limit )
        self.sound_prediction.clear_histogram()

    def process_data_msg(self, msg):
        if type(msg) == AudioftsMessage:
            #self.logger.info(f"Soundclass processing started")
            timestamps = []
            valids = []
            classes = []
            for i in range(len(msg.timestamp)):
                valid, sound_class = self.sound_prediction.sound_class(msg.feature[i])
                valids.append(valid)
                classes.append(sound_class)
                timestamps.append(msg.timestamp[i])
                #print(f"Sound feature {msg.feature[i]} class {sound_class}")
            return SoundclassMessage(timestamps, valids, classes)


def soundclass(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Soundclass(config, status_uri, data_in_uris, data_out_ur)
    print(f"Soudclass started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Soundclass")
    sleep(0.5)
    exit()