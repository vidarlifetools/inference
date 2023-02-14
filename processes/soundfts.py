# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.sound import SoundMessage
from sound_utils import sound_feature
from utilities.ring_buffer import ring_buffer
import time

MODULE_SOUNDFTS = "Soundfts"

@dataclass
class SoundftsMessage:
    timestamp: list = None
    valid: list = True
    feature: list = None


@dataclass
class SoundftsConfig:
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


class Soundfts(DataModule):
    name = MODULE_SOUNDFTS
    config_class = SoundftsConfig

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
        if type(msg) == SoundMessage:
            #self.logger.info(f"Soundfts processing started")
            self.ring_buffer.put(msg.samples)
            if self.ring_buffer.get_length() > self.audio_buffer_size:
                self.ring_buffer.get(self.audio_buffer, self.audio_buffer_size, self.audio_buffer_step)
                return SoundftsMessage(msg.timestamp, True, self.sound_feature.get_feature(self.audio_buffer))


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