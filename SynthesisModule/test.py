from __future__ import print_function, division, absolute_import

import platform
# TrungNT: Sr, my Mac require doing this
if "Darwin" in platform.platform():
    import matplotlib
    matplotlib.use("TkAgg")

import os
import numpy as np
from scipy import stats

from utils import Audio, Video, read_openface
from expression import Happy, Sad, Laugh, interpolate


laugh = Laugh(2, 300)
happy = Happy(1, 76)
sad = Sad(1, 96)

happy.set_reference(laugh[0]).after(laugh)
sad.set_reference(laugh[0]).after(happy)


# ===========================================================================
# Example video player
# This example play n9.wav whenever the farmes end
# ===========================================================================
b = Video()
b.play_expression(laugh)
b.play_frames(happy.frames).play_frames(sad.frames)
b.save('/Users/trungnt13/tmp/enterface', keep_cache=True)
