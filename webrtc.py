import asyncio
import json
import logging
import time
from typing import Tuple, Union
from av.frame import Frame
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 25fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

from aiortc import MediaStreamTrack

logging.basicConfig()
logger = logging.getLogger(__name__)


class PlayerStreamTrack(MediaStreamTrack):
    """
    A media stream track that reads frames from a player.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self._start = None
        self._timestamp = None
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if self._timestamp is not None:
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                print('video start:', self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else:  # audio
            if self._timestamp is not None:
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                print('audio start:', self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame]:
        self._player._start(self)
        frame = await self._queue.get()
        if frame is None:
            self.stop()
            raise Exception('No more frames')

        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                print(f"------actual avg final fps: {self.framecount / self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
        return frame

    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


class HumanPlayer:
    def __init__(self, nerfreal):
        self.__started = set()
        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        self.__container = nerfreal
        self.__task = None
        self.__quit_event = asyncio.Event()

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A MediaStreamTrack instance if the player has audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A MediaStreamTrack instance if the player has video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__task is None:
            self.__log_debug("Starting worker task")
            self.__task = asyncio.create_task(self.player_worker())

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__task is not None:
            self.__log_debug("Stopping worker task")
            self.__quit_event.set()
            self.__task = None

        if not self.__started and self.__container is not None:
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"HumanPlayer {msg}", *args)

    async def player_worker(self):
        await self.__container.render(self.__quit_event, self.__audio, self.__video)
        self.__log_debug("Player worker stopped")

    def put_audio_frame(self, frame_data: np.ndarray):
        frame = AudioFrame.from_ndarray(frame_data, layout='mono', format='s16')
        frame.sample_rate = SAMPLE_RATE
        if self.__audio.readyState == "live":
            asyncio.create_task(self.__audio._queue.put(frame))

    def put_video_frame(self, frame: Frame):
        if self.__video.readyState == "live":
            asyncio.create_task(self.__video._queue.put(frame))
