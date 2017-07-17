from __future__ import print_function, division, absolute_import

import os
import sys
from six import add_metaclass

from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np

from utils import read_openface

# ===========================================================================
# Helper
# ===========================================================================
# path to the database
DATABASE = os.path.join(os.path.dirname(sys.argv[0]), "database")


def _searching_expression(name, level, nframes, extension):
    def distance_func(x):
        x = x.split('.')[0].split('_')
        l1 = int(x[1])
        n1 = int(x[2])
        return abs(l1 - level) * 10000 + abs(n1 - nframes)
    all_files = [i for i in os.listdir(DATABASE) if extension in i]
    # match the name
    all_files = [i for i in all_files if name + "_" in i]
    # match all <= intensity, and sorted by order of decreasing
    all_files = sorted([i for i in all_files if int(i.split('_')[1]) <= level],
                       key=lambda x: distance_func(x))
    if len(all_files) == 0:
        raise ValueError("Cannot file any file with, "
            "name:%s, level:%d, nframes:%d, ext:%s" % (name, level, nframes, extension))
    return os.path.join(DATABASE, all_files[0])


# ===========================================================================
# Expression
# ===========================================================================
@add_metaclass(ABCMeta)
class Expression(object):
    """ Expression
    Attributes
    ----------
    frames: ndarray [nb_frames x nb_points_per_frames(68) x 3(x,y,z)]
    audio: str (path to wav file)
    """

    def __init__(self, level, nframes):
        super(Expression, self).__init__()
        self._name = self.__class__.__name__.lower()
        self._level = level
        self._nframes = nframes
        self._frames = read_openface(_searching_expression(
            self._name, self._level, self._nframes, '.csv'))
        self._audio = _searching_expression(
            self._name, self._level, self._nframes, '.wav')

    def __getitem__(self, idx):
        return self._frames[idx]

    @property
    def name(self):
        return "%s_%d_%d" % (self._name, self._level, self._nframes)

    @property
    def level(self):
        return self._level

    @property
    def nframes(self):
        return self._nframes

    @property
    def frames(self):
        return self._frames

    @property
    def audio(self):
        return self._audio

    def __str__(self):
        return "<[%s] level:%d #frames:%d audio:%s>" % \
        (self.__class__.__name__, self._level, self._nframes, self._audio)

    # ==================== Frames manipulation ==================== #
    def concat(self, last_frame):
        offset = self._frames[1:] - self._frames[:-1]
        offset = np.cumsum(offset, axis=0)
        last_frame = np.expand_dims(last_frame, 0)
        return offset + last_frame

    def merge(self, expr):
        pass

    def length(self, nframes):
        pass

    def intensity(self, level):
        pass

# NOTE: you only need to name the class matching the expression name.


class Happy(Expression):

    pass


class Sad(Expression):

    pass
