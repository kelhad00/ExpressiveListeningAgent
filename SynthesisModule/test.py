from __future__ import print_function, division, absolute_import

import os
import numpy as np

from utils import Audio, Video, read_openface
# from face3danim import get3d_data
from expression import Expression

b = Video()
# data = get3d_data('database/happy_1_77.csv')
data1 = read_openface('database/happy_1_77.csv')
data2 = read_openface('database/happy_1_96.csv')

b.play_frames(data1).play_frames(data2)
b.run()
