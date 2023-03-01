# import time
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.soundfts import SoundftsMessage
from sound_utils import SoundPrediction
import time
from constants import *

MODULE_SOUNDCLASS = "Soundclass"

@dataclass
class SoundclassMessage:
    timestamp: list = None
    valid: list = None
    sound_class: list = None
    frame_no: int = -1


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

        self.sound_prediction = SoundPrediction(
            self.config.model_filename_split_sound,
            self.config.model_filename_client,
            self.config.model_filename_sound,
            self.config.histogram_depth,
            self.config.histogram_limit )
        self.sound_prediction.clear_histogram()

    def process_data_msg(self, msg):
        if type(msg) == SoundftsMessage:
            frame_no = msg.frame_no
            if msg.valid:
                #self.logger.info(f"Soundclass processing started")
                valid, sound_class = self.sound_prediction.get_class(msg.feature)
                #print(f"Sound feature {msg.feature[i]} class {sound_class}")
                return SoundclassMessage(msg.timestamp, valid, sound_class, frame_no)
            else:
                return SoundclassMessage(msg.timestamp, False, MISSING_CLASS_SOUND, frame_no)


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