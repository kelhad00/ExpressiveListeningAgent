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
    # ====== found file ====== #
    found = all_files[0]
    desire_name = "%s_%d_%d" % (name, level, nframes)
    if found.split('.')[0] != desire_name:
        print("[WARNING] Request expression with configuration: %-12s, but only "
            "found expression: %-12s" % (desire_name, found.split('.')[0]))
    return os.path.join(DATABASE, found)


def interpolate(last_frame, first_frame, nFrames):
    res = []
    for i in range(nFrames):
        temp = [(1 - i / float(nFrames - 1)) * x + (i / float(nFrames - 1)) * y
                for x, y in zip(last_frame, first_frame)]
        temp = np.concatenate([t[None, :] for t in temp])[None, :]
        res.append(temp)
    res = np.concatenate(res, 0)
    return res


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
        # ====== find appropriate frames file ====== #
        _ = _searching_expression(self._name, self._level, self._nframes, '.csv')
        self._frames = read_openface(_)
        # infer audio file name from frames file name
        _ = os.path.basename(_).split('.')[0] + '.wav'
        self._audio = os.path.join(DATABASE, _)
        if not os.path.exists(self._audio):
            raise RuntimeError("Cannot find audio file for the expression: %s, "
                "at path: %s" % (self.name, self._audio))

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

    def copy(self):
        clazz = self.__class__
        obj = clazz.__new__(clazz)
        obj._name = self._name
        obj._level = self._level
        obj._nframes = self._nframes
        obj._frames = np.copy(self._frames)
        obj._audio = self._audio
        return obj

    def __str__(self):
        return "<[%s] level:%d #frames:%d audio:%s>" % \
        (self.__class__.__name__, self._level, self._nframes, self._audio)

    # ==================== Frames manipulation ==================== #
    def set_reference(self, frame):
        """ Use a frame as a reference for this expression
        * Calculating the cummulative differences of all frames to the first
        frame of this expression.
        * Adding the differences to the reference frame.
        """
        frame = np.expand_dims(frame, 0)
        if frame.shape != (1,) + self.frames.shape[1:]:
            raise ValueError("Reference frame must have shape: %s" %
                self.frames.shape[1:])
        offset = self._frames[1:] - self._frames[:-1]
        offset = np.cumsum(offset, axis=0)
        self._frames = frame + offset
        return self

    def concat(self, last_frame, interp=5):
        """
        """
        # ====== interpolate first ====== #
        interp = interpolate(last_frame, self.frames[0], nFrames=interp)
        # ====== adding offset ====== #
        offset = self._frames[1:] - self._frames[:-1]
        offset = np.cumsum(offset, axis=0)
        last_frame = np.expand_dims(interp[-1], 0)
        offset = offset + last_frame
        # ====== assign ====== #
        self._frames = np.concatenate([interp, offset], axis=0)
        return self

    def merge(self, expr):
        raise NotImplementedError

    def adjust(self, length, nframes):
        raise NotImplementedError


# NOTE: you only need to name the class matching the expression name.
class Happy(Expression):
    pass


class Sad(Expression):
    pass


class Laugh(Expression):
    pass
