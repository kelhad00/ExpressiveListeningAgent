from __future__ import print_function, division, absolute_import

import os
from six import add_metaclass

from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np

# path to the database
DATABASE = "database"


@add_metaclass(ABCMeta)
class Expression(object):
    """ Expression """

    def __init__(self, name, level, nframes):
        super(Expression, self).__init__()
        video_files = [i for i in os.listdir(DATABASE) if '.csv' in i]
        audio_files = [i for i in os.listdir(DATABASE) if '.wav' in i]
        print(video_files, audio_files)
