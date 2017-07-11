# ===========================================================================
# Require packages:
# * pyaudio
# * numpy
# * pandas
# ===========================================================================
from __future__ import print_function, division, absolute_import

# TrungNT: Sr, my Mac require doing this
import platform
if "Darwin" in platform.platform():
    import matplotlib
    matplotlib.use("TkAgg")

import numpy as np
# for audio
import wave
import pyaudio
# for video
from matplotlib import animation
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


# ===========================================================================
# Audio manipulation
# ===========================================================================
CHUNK_SIZE = 1024


class AudioData(object):

    def __init__(self, path):
        super(AudioData, self).__init__()
        wf = wave.open(path, 'rb')
        sr, s = wf.getframerate(), wf.readframes(wf.getnframes())
        if wf.getnchannels() > 1:
            raise ValueError("Only support 1-channel wav")
        self.raw = s
        self.sr = sr
        self.sample_width = wf.getsampwidth()
        self.channels = wf.getnchannels()
        wf.close()

        self._path = path
        self._counter = 0

    def __len__(self):
        return len(self.raw) - self._counter

    def validate(self, sample_width, sr, channels):
        if self.sr != sr:
            raise ValueError("invalid sample rate %d != %d" % (self.sr, sr))
        if sample_width != self.sample_width:
            raise ValueError("invalid sample width %d != %d" %
                (self.sample_width, sample_width))
        if channels != self.channels:
            raise ValueError("invalid number of channels %d != %d" %
                (self.channels, channels))
        return self

    def read(self, frame_count):
        data = self.raw[self._counter: self._counter + frame_count * self.sample_width]
        self._counter += (frame_count * self.sample_width)
        self._counter = min(self._counter, len(self.raw))
        return data

    @property
    def ended(self):
        return self._counter == len(self.raw)

    def __str__(self):
        return "<%s sr:%d nframes:%d counter:%d sample_width:%d>" % \
            (self._path, self.sr, len(self.raw), self._counter, self.sample_width)


class Audio(object):
    """
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
        self.pya = pyaudio.PyAudio()
        self._stream = None
        self._audio = []
        # ====== audio info ====== #
        self.sample_width = None # 2 <=> int16
        self.sr = None # sample rate
        self.channels = None
        # ====== status flags ====== #
        self._is_stream_stopped = False
        self._is_terminated = False
        self._callback_end = lambda *args, **kwargs: None

    def set_callback(self, callback):
        """ This callback is called when the data is exhausted. """
        if not callable(callback):
            raise ValueError("`callback` must be callable.")
        self._callback_end = callback

    @property
    def data(self):
        return self._audio

    def play(self, path):
        if self._is_terminated:
            raise RuntimeError("The player is terminated.")
        # ====== stream the audio ====== #
        a = AudioData(path)
        if self.sample_width is None:
            self.sample_width = a.sample_width
            self.sr = a.sr
            self.channels = a.channels
        self._audio.append(
            a.validate(self.sample_width, self.sr, self.channels))
        # ====== create stream if not available ====== #
        if self._stream is None:
            self._stream = self.pya.open(
                format=self.pya.get_format_from_width(self.sample_width),
                channels=self.channels,
                rate=self.sr,
                frames_per_buffer=CHUNK_SIZE,
                output=True,
                start=False,
                stream_callback=lambda in_data, frame_count, time_info, status:
                    self._callback(in_data, frame_count, time_info, status))
        # ====== restart or start the stream if necessary ====== #
        if (self._is_stream_stopped or self._stream.is_stopped()) and \
        len(self.data) > 0:
            self._stream.stop_stream()
            self._stream.start_stream()
        return self

    def _extract_frames(self, frame_count):
        data = ""
        N = frame_count
        while N > 0 and len(self._audio) > 0:
            audio = self._audio[-1]
            new_data = audio.read(N)
            data = data + new_data
            N -= len(new_data) // audio.sample_width
            if audio.ended: # remove audio if finished reading
                self._audio.pop()
        return data

    def _callback(self, in_data, frame_count, time_info, status):
        data = self._extract_frames(frame_count)
        if len(data) // self.sample_width != frame_count:
            self._is_stream_stopped = True
            self._callback_end()
        return (data, pyaudio.paContinue)

    def terminate(self):
        # stop stream (4)
        self._stream.stop_stream()
        self._stream.close()
        # close PyAudio (5)
        self.pya.terminate()
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
        self._frames = []
        self._animation = None
        self._fps = 30
        self._callback_frames = lambda *args, **kwargs: None
        self._audio_player = Audio()

    def set_callback(self, callback_frames=None, callback_audio=None):
        """ This callback is called when the data is exhausted. """
        if callback_frames is not None:
            if not callable(callback_frames):
                raise ValueError("`callback_frames` must be callable.")
            self._callback_frames = callback_frames
        if callback_audio is not None:
            if not callable(callback_audio):
                raise ValueError("`callback_audio` must be callable.")
            self._audio_player.set_callback(callback_audio)

    @property
    def data_frames(self):
        return self._frames

    @property
    def data_audio(self):
        return self._audio_player.data

    def play_frames(self, data):
        # ====== check data ====== #
        if data.ndim != 3 and data.shape[1] != NB_POINTS_PER_FRAMES:
            raise ValueError("`data` must has shape (nb_frames, 68, 3).")
        for x in data:
            self._frames.append(x)
        return self

    def play_audio(self, path):
        self._audio_player.play(path)

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
                interval=int(1000. / self._fps),
                repeat=False,
                repeat_delay=None,
                blit=True)
            plt.ioff(); plt.show(block=False)

    def _update(self, lines):
        # get the frames
        if len(self._frames) > 0:
            for line, point in zip(lines, self._frames.pop()):
                line.set_data(point[:2])
                line.set_3d_properties(point[-1:])
            # play audio every 20 frames
            if len(self._frames) % 20 == 0:
                self.play_audio('data/n9.wav')
        else:
            self._callback_frames()
        return lines

    def terminate(self):
        # stop stream (4)
        self._audio_player.terminate()
