# import time
import numpy as np
from dataclasses import dataclass
from framework.module import DataModule
from time import sleep
from processes.camera import CameraMessage
from processes.compound import CompoundMessage
import cv2
import time

MODULE_PRESENT = "Present"

@dataclass
class PresentMessage:
    fps: int = 15
    image: np.array = None


@dataclass
class PresentConfig:
    name: str = ""
    scale: float = 1.0


class Present(DataModule):
    name = MODULE_PRESENT
    config_class = PresentConfig

    def __init__(self, *args):
        super().__init__(*args)
        self.current_class = 0

    def process_data_msg(self, msg):
        if type(msg) == CameraMessage:
            # resize image
            width = int(msg.image.shape[1] * self.config.scale)
            height = int(msg.image.shape[0] * self.config.scale)
            dim = (width, height)
            resized = cv2.resize(msg.image, dim, interpolation=cv2.INTER_AREA)

            im_shape = resized.shape
            bottom_image = np.zeros((100, im_shape[1], im_shape[2]), dtype=type(msg.image[0,0,0]))
            cv2.putText(bottom_image,str(self.current_class),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.concatenate((resized, bottom_image))
            cv2.imshow("Final image", img)
            cv2.waitKey(1)
        if type(msg) == CompoundMessage:
            self.current_class = msg.compound_class
            # resize image
        return None


def present(start, stop, config, status_uri, data_in_uris, data_out_ur):
    proc = Present(config, status_uri, data_in_uris, data_out_ur)
    print(f"Present started at {time.time()}")
    while not start.is_set():
        sleep(0.1)
    proc.start()
    while not stop.is_set():
        sleep(0.1)
    proc.stop()
    print("Ending Present")
    sleep(0.5)
    exit()