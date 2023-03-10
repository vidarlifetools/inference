# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
import soundfile as sf
import time

MODULE_SOUND = "Sound"

@dataclass
class SoundMessage:
    timestamp: float = 0.0
    valid: bool = True
    sr: int = 16000
    samples: np.array = None


@dataclass
class SoundConfig:
    name: str = ""
    sound_path: str = ""


class Sound(DataModule):
    name = MODULE_SOUND
    config_class = SoundConfig

    def __init__(self, *args):
        super().__init__(*args)
        sound_file = self.config.sound_path
        self.samples, self.sr = sf.read(sound_file)
        if len(self.samples.shape) > 1:
            self.samples = self.samples[:,0]
        self.sound_idx = 0

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            #self.logger.info(f"Sound processing started")
            # Create a buffer with audio samples covering time between frames
            no_of_samples = int(self.sr / msg.fps)
            if self.sound_idx + no_of_samples < len(self.samples):
                sound_buffer = self.samples[self.sound_idx:self.sound_idx+no_of_samples]
                self.sound_idx += no_of_samples
                return SoundMessage(msg.timestamp, True, self.sr, sound_buffer)
        #else:
        #    return AudioMessage(False, self.sr, None)


def sound(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Sound(config, status_uri, data_in_uris, data_out_ur)
    print(f"Sound started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Sound")
    sleep(0.5)
    exit()