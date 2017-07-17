from __future__ import print_function, division, absolute_import

import os
import numpy as np

from utils import Audio, Video, read_openface
# from face3danim import get3d_data
from expression import Happy, Sad, Laugh

happy = Happy(1, 76)
sad = Sad(1, 96)
laugh = Laugh(2, 300)
last_exp = sad


# ===========================================================================
# Example video player
# This example play n9.wav whenever the farmes end
# ===========================================================================
def logic(video):
    # you should see shit continuosly appear on screen
    # print("Shit", video.last_name)
    pass


def callback(video):
    global last_exp
    frames = last_exp.concat(video.last_frame)
    # flip the expression
    last_exp = happy if last_exp == sad else happy
    video.play_frames(frames, queue=True)
    video.audio.play("data/n9.wav")
    # print("END")

b = Video()
b.play_expression(laugh)
b.set_callback(end_frames=callback, logic_processing=logic)
b.run()
# b.terminate()
