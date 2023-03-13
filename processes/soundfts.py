# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.sound import SoundMessage
from sound_utils import SoundFeature
from utilities.ring_buffer import ring_buffer
import time
from feature_constants import\
    sound_sample_rate,\
    sound_feature_length,\
    sound_feature_step


MODULE_SOUNDFTS = "Soundfts"

@dataclass
class SoundftsMessage:
    timestamp: list = None
    valid: list = True
    feature: list = None
    frame_no: int = -1


@dataclass
class SoundftsConfig:
    name: str = ""

class Soundfts(DataModule):
    name = MODULE_SOUNDFTS
    config_class = SoundftsConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.ring_buffer = ring_buffer(sound_sample_rate)
        self.audio_buffer_size = int(sound_feature_length*sound_sample_rate)
        self.audio_buffer_step = int(sound_feature_step*sound_sample_rate)
        self.audio_buffer = np.zeros((self.audio_buffer_size,), dtype=float)

        self.sound_feature = SoundFeature()

    def process_data_msg(self, msg):
        if type(msg) == SoundMessage:
            frame_no = msg.frame_no

            self.logger.debug(f"Soundfts processing started for frame {frame_no}")
            self.ring_buffer.put(msg.samples)
            if self.ring_buffer.get_length() > self.audio_buffer_size:
                self.ring_buffer.get(self.audio_buffer, self.audio_buffer_size, self.audio_buffer_step)
                return SoundftsMessage(msg.timestamp, True, self.sound_feature.get_feature(self.audio_buffer), frame_no)
            else:
                return SoundftsMessage(msg.timestamp, False, None, frame_no)


def soundfts(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Soundfts(config, status_uri, data_in_uris, data_out_ur)
    print(f"Soundfts started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Soundfts")
    sleep(0.5)
    exit()