# ===========================================================================
# Require packages:
# * wave
# * numpy
# * pandas
# ===========================================================================
from __future__ import print_function, division, absolute_import

# TrungNT: Sr, my Mac require doing this
import platform
if "Darwin" in platform.platform():
    import matplotlib
    matplotlib.use("TkAgg")
import os
import sys
import time
from six import string_types

import numpy as np
# for audio
from pygame import mixer
# for video
from matplotlib import animation
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


# ===========================================================================
# Audio manipulation
# ===========================================================================
class Audio(object):
    """
    Parameters
    ----------
    backend: str
        audio

    Example
    -------
    >>> player = Audio()
    >>> for i in range(11):
    >>>     player.play('data/audio%d.wav' % i)
    >>> for i in range(11):
    >>>     player.play('data/audio%d.wav' % i)
    >>> while len(player.audio) > 0:
    >>>     time.sleep(0.1)
    """

    def __init__(self):
        super(Audio, self).__init__()
        self._audio = {}
        self._channel = None
        # ====== status flags ====== #
        self._is_initialized = False
        self._is_terminated = False

    @property
    def data(self):
        return self._audio

    def _on_event(self, event):
        print("shit")

    def play(self, path, loops=0, maxtime=0, fade_ms=0, volume=1.,
             queue=True):
        if self._is_terminated:
            raise RuntimeError("The player is terminated.")
        path = os.path.abspath(path)
        # ====== check initialized ====== #
        if not self._is_initialized:
            self._is_initialized = True
            mixer.init()
            self._channel = mixer.find_channel()
            self._channel.set_endevent(12082518) # beautiful number
        # ====== load the audio ====== #
        if path not in self._audio:
            self._audio[path] = mixer.Sound(path)
        path = self._audio[path]
        path.set_volume(volume)
        # ====== play in the channels ====== #
        if queue:
            self._channel.queue(path)
        else:
            self._channel.play(path, loops, maxtime, fade_ms)
        return self

    def get_playing(self):
        s = self._channel.get_sound()
        for i, j in self._audio.iteritems():
            if j == s:
                return i
        return None

    def fadeout(self, ms):
        self._channel.fadeout(ms)
        return self

    def stop(self, path=None):
        if path is not None:
            path = os.path.abspath(path)
            self._audio[path].stop()
        else:
            self._channel.stop()
        return self

    def pause(self, path=None):
        if path is not None:
            path = os.path.abspath(path)
            self._audio[path].pause()
        else:
            self._channel.pause()
        return self

    def resume(self, path=None):
        if path is not None:
            path = os.path.abspath(path)
            self._audio[path].unpause()
        else:
            self._channel.unpause()
        return self

    def terminate(self):
        mixer.quit()
        self._is_terminated = True


# ===========================================================================
# Video player
# ===========================================================================
NB_POINTS_PER_FRAMES = 68


def read_openface(f):
    """Return a numpy array with shape:
    # nb_frames x nb_points_per_frames(68) x 3(x,y,z)
    """
    import pandas as pd
    raw_data = pd.read_csv(f)
    #find the location of 3d landmarks
    a = raw_data.columns.get_loc(' X_0')
    b = raw_data.columns.get_loc(' Z_67')
    nb_points = (b + 1 - a) // 3 # nb of points for each frames
    data = raw_data.ix[:, a:b + 1].as_matrix() # nb_frames x nb_points
    # nb_frames x nb_points_per_frames x 3
    data = np.swapaxes(data.reshape(data.shape[0], 3, nb_points), 1, 2)
    # switch the position of y and z to have a right view
    data = np.concatenate([data[:, :, 0][:, :, None],
                           data[:, :, 2][:, :, None],
                           data[:, :, 1][:, :, None]], axis=-1)
    # To get the first frames: data[0]
    # to get the first point of first frames: data[0][0]
    return data


class Video(object):

    def __init__(self, fps=30):
        super(Video, self).__init__()
        # ====== control properties ====== #
        self._frames = []
        self._animation = None
        self._spf = 1. / fps # second per frame
        self._audio_player = Audio()
        # ====== internal states ====== #
        self._curr_time = 0
        # ====== all callback function ====== #
        self._end_frames = lambda video: None
        self._logic_processing = lambda video: None
        self._last_frame = None

    def set_callback(self, end_frames=None, logic_processing=None):
        """ This callback is called when the data is exhausted. """
        if end_frames is not None and callable(end_frames):
            self._end_frames = end_frames
        if logic_processing is not None and callable(logic_processing):
            self._logic_processing = logic_processing
        return self

    @property
    def last_frame(self):
        return self._last_frame

    @property
    def data_frames(self):
        return self._frames

    @property
    def data_audio(self):
        return self._audio_player.data

    @property
    def audio(self):
        return self._audio_player

    def play_frames(self, data, queue=True):
        # ====== check data ====== #
        if data.ndim != 3 and data.shape[1] != NB_POINTS_PER_FRAMES:
            raise ValueError("`data` must has shape (nb_frames, 68, 3).")
        if not queue:
            self._frames = []
        for x in data:
            self._frames.append(x)
        return self

    def run(self):
        """Given the fact that we try hard, this method still blocking after
        we call plt.show"""
        if len(self.data_frames) == 0:
            raise RuntimeError("No frames data found!")
        if self._animation is None:
            def infinite_iterator():
                while True: yield 8
            fig = plt.figure(figsize=(6, 8), dpi=160)
            ax = p3.Axes3D(fig)
            ax.axis('off')
            ax.grid(False)
            ax.view_init(-10, 85)
            ax.set_title('Avatar')
            ax.set_zlim3d([100, -60])
            ax.disable_mouse_rotation()
            # create dummy point
            lines = [ax.plot([x], [y], [z], 'k.', animated=True)[0]
                     for x, y, z in self._frames.pop()]
            self._animation = animation.FuncAnimation(fig=fig,
                func=lambda num, lines: self._update(lines),
                frames=infinite_iterator(),
                fargs=(lines,),
                interval=1, # 1ms delay
                repeat=False,
                repeat_delay=None,
                blit=True)
            plt.ioff(); plt.show(block=False)

    def _update(self, lines):
        # ====== processing the logic ====== #
        self._logic_processing(self)
        # ====== update frames ====== #
        new_time = time.time()
        if new_time - self._curr_time >= self._spf:
            self._curr_time = new_time
            # get the frames
            if len(self._frames) > 0:
                self._last_frame = self._frames.pop()
                for line, point in zip(lines, self._last_frame):
                    line.set_data(point[:2])
                    line.set_3d_properties(point[-1:])
            else:
                self._end_frames(self)
        return lines

    def terminate(self):
        # stop stream (4)
        self._audio_player.terminate()
