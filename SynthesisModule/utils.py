# ===========================================================================
# Require packages:
# * pygame (for playing audio)
# * numpy
# * soundfile (for saving audio)
# * pandas
# * ffmpeg (for converting gif + wav => mp4)
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
from pygame import mixer
import soundfile as sf

# for video
from matplotlib import animation
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


# ===========================================================================
# Audio manipulation
# ===========================================================================
def resample_audio(s, fs_orig, fs_new):
    '''
    '''
    fs_orig = int(fs_orig)
    fs_new = int(fs_new)
    if fs_new > fs_orig:
        raise ValueError("Do not support upsampling audio from %d(Hz) to %d(Hz)."
            % (fs_orig, fs_new))
    elif fs_orig != fs_new:
        import resampy
        s = resampy.resample(s.astype('float32'), sr_orig=fs_orig, sr_new=fs_new)
    return s


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

    def __init__(self, fps=30, logging=True):
        super(Video, self).__init__()
        # ====== control properties ====== #
        self._frame_id = 0
        self._frames = [] # list of [(name, frames), ...]
        self._animation = None
        self._spf = 1. / fps # second per frame
        # ====== audio ====== #
        self._audio_player = Audio()
        self._audio_map = {} # mapping expression name to audio path
        # ====== internal states and tracking ====== #
        self._curr_time = 0
        self._curr_frames = None
        self._last_frame = None
        self._last_name = None
        # ====== all callback function ====== #
        self._end_frames = lambda video: None
        self._logic_processing = lambda video: None
        # ====== logging ====== #
        self.logging = logging

    def set_callback(self, end_frames=None, logic_processing=None):
        """ This callback is called when the data is exhausted. """
        if end_frames is not None and callable(end_frames):
            self._end_frames = end_frames
        if logic_processing is not None and callable(logic_processing):
            self._logic_processing = logic_processing
        return self

    # ==================== Current states ==================== #
    @property
    def last_frame(self):
        return self._last_frame

    @property
    def last_name(self):
        return self._last_name

    @property
    def curr_frames(self):
        return self._curr_frames

    # ==================== Attributes ==================== #
    @property
    def data_frames(self):
        return self._frames

    @property
    def data_audio(self):
        return self._audio_player.data

    @property
    def audio(self):
        return self._audio_player

    # ==================== controller ==================== #
    def play_frames(self, data, name=None, queue=True):
        # ====== check data ====== #
        if data.ndim != 3 and data.shape[1] != NB_POINTS_PER_FRAMES:
            raise ValueError("`data` must has shape (nb_frames, 68, 3).")
        # ====== check name ====== #
        self._frame_id += 1
        if name is None:
            name = "Frame%d" % self._frame_id
        else:
            name = str(name)
        # ====== add frames ====== #
        if not queue:
            self._frames = []
        # reverse the frames order
        self._frames.append((name, [x for x in data][::-1]))
        return self

    def play_expression(self, exp):
        # reverse the frames order
        frames = [x for x in exp.frames][::-1]
        audio = exp.audio
        self._frames.append((exp.name, frames))
        self._audio_map[exp.name] = audio
        return self

    # ==================== Running ==================== #
    def run(self):
        """Given the fact that we try hard, this method still blocking after
        we call plt.show"""
        if len(self._frames) == 0 and len(self._frames[-1][-1]) == 0:
            raise RuntimeError("No frames data found!")
        # assign first current frames
        self._curr_frames = self._frames.pop()
        name = self._curr_frames[0]
        if name in self._audio_map:
            self._log("Audio", "Playing %s" % (self._audio_map[name]))
            self._audio_player.play(self._audio_map[name])
        # create the animation
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
                     for x, y, z in self._curr_frames[-1].pop()]
            self._animation = animation.FuncAnimation(fig=fig,
                func=lambda num, lines: self._update(lines),
                frames=infinite_iterator(),
                fargs=(lines,),
                interval=1, # 1ms delay
                repeat=False,
                repeat_delay=None,
                blit=True)
            plt.ioff(); plt.show(block=True)

    def save(self, path, dpi=None, sr=8000, mismatch_threshold=0.5,
             keep_cache=False):
        """
        mismatch_threshold: float, the threshold for difference between
            audio and video duration (in second)
        """
        path = path.split(".")[0]
        if len(self._frames) == 0 and len(self._frames[-1][-1]) == 0:
            raise RuntimeError("No frames data found!")
        # ====== get all frames and audio ====== #
        frames = []
        audio = []
        duration = [] # in second
        for name, f in self._frames[::-1]:
            frames += f
            audio.append(None if name not in self._audio_map
                        else self._audio_map[name])
            duration.append(len(f) * self._spf)
        # reset the internal frames
        self._frames = []
        # ====== save the audio ====== #
        audio = [(np.zeros(shape=(int(length * sr),)), sr) if i is None
                 else sf.read(i)
                 for i, length in zip(audio, duration)]
        audio = [resample_audio(i, fs_orig=j, fs_new=sr) for i, j in audio]
        # valiate audio duration match frames duration
        for y, frame_dur in zip(audio, duration):
            audio_dur = len(y) / sr
            if np.abs(audio_dur - frame_dur) > mismatch_threshold:
                raise RuntimeError("Audio and video duratoin mismatch: %.2f != %.2f"
                    % (audio_dur, frame_dur))
        # saving all the audio
        self._log('Save', 'Saving audio file at path: %s.wav' % path)
        with sf.SoundFile(file=path + '.wav', mode='w',
                          samplerate=sr, channels=1) as f:
            # have to reverse the audio to have the same order with the video
            f.write(np.concatenate(audio[::-1]))
        # ====== save all frames using Animation ====== #
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
                 for x, y, z in frames.pop()]
        N = len(frames)

        def update(num, lines):
            if len(frames) > 0:
                if (N - len(frames)) % 25 == 0:
                    self._log('Saved', '%d/%d frames' % (N - len(frames), N))
                for line, point in zip(lines, frames.pop()):
                    line.set_data(point[:2])
                    line.set_3d_properties(point[-1:])
            else:
                self._log('Finnished', '%d/%d frames' % (N, N))
            return lines

        ani = animation.FuncAnimation(fig=fig,
            func=update,
            frames=range(N),
            fargs=(lines,),
            interval=self._spf * 1000, # delay in millisecond
            repeat=False,
            repeat_delay=None,
            blit=True,
            save_count=N)
        self._log('Save', 'Saving animation at path: %s.gif' % path)
        ani.save(path + '.gif', writer='imagemagick', fps=1. / self._spf, dpi=dpi)
        # ====== merge audio and frames into mp4 ====== #
        self._log('Save', 'Merging audio and frames into mp4 file: %s.mp4' % path)
        try:
            os.remove('%s.mp4' % path)
        except Exception:
            pass
        os.system('ffmpeg -i %s.gif -i %s.wav -strict experimental %s.mp4' %
            (path, path, path))
        # ====== remove unecessary file ====== #
        if not keep_cache:
            try:
                os.remove('%s.gif' % path)
                os.remove('%s.wav' % path)
            except Exception:
                pass

    def _update(self, lines):
        new_time = time.time()
        # ====== processing the logic ====== #
        self._logic_processing(self)
        # ====== update frames ====== #
        if new_time - self._curr_time >= self._spf:
            # print(new_time - self._curr_time)
            curr_fps = 1. / (new_time - self._curr_time)
            self._curr_time = new_time
            # callback end_frames
            if len(self._curr_frames[-1]) == 0:
                if len(self._frames) == 0:
                    self._log("Video", "End of frames")
                    self._end_frames(self)
                    return lines
                else: # start new expression
                    self._curr_frames = self._frames.pop()
                    name = self._curr_frames[0]
                    if name in self._audio_map:
                        self._log("Audio", "Playing %s" % (self._audio_map[name]))
                        self._audio_player.play(self._audio_map[name])
            # processing frames
            name, frames = self._curr_frames
            self._log("Video", "Playing %s, remain: %d, fps: %.2f, c-fps: %.2f" %
                (name, len(frames), 1. / self._spf, curr_fps))
            # store last states
            self._last_frame = frames.pop()
            self._last_name = name
            # set new frames
            for line, point in zip(lines, self._last_frame):
                line.set_data(point[:2])
                line.set_3d_properties(point[-1:])
        return lines

    def _log(self, tag, msg):
        if self.logging:
            print("[Player] (%s): %s" % (tag, msg))

    def terminate(self):
        # stop stream (4)
        self._audio_player.terminate()
