import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import AsyncIterator

import aiohttp

from io import BytesIO
from enum import Enum

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    def __init__(self, opt, parent):
        self.opt = opt
        self.parent = parent

        self.fps = opt.fps  # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = asyncio.Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        # Clear the queue
        while not self.msgqueue.empty():
            self.msgqueue.get_nowait()
        self.state = State.PAUSE

    def put_msg_txt(self, msg):
        self.msgqueue.put_nowait(msg)

    def render(self, quit_event):
        # Start the process_tts coroutine as an asyncio Task
        asyncio.create_task(self.process_tts(quit_event))

    async def process_tts(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = await asyncio.wait_for(self.msgqueue.get(), timeout=1)
                self.state = State.RUNNING
                await self.txt_to_audio(msg)
            except asyncio.TimeoutError:
                continue
        print('ttsreal task stopped')

    async def txt_to_audio(self, msg):
        pass

###########################################################################################
class EdgeTTS(BaseTTS):
    async def txt_to_audio(self, msg):
        voicename = "zh-CN-YunxiaNeural"
        text = msg
        t = time.time()
        await self.__main(voicename, text)
        print(f'-------edge tts time: {time.time() - t:.4f}s')

        self.input_stream.seek(0)
        stream = await self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            self.parent.put_audio_frame(stream[idx:idx + self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate()

    async def __create_bytes_stream(self, byte_stream):
        loop = asyncio.get_event_loop()
        stream, sample_rate = await loop.run_in_executor(None, sf.read, byte_stream)
        print(f'[INFO] tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = await loop.run_in_executor(None, resampy.resample, stream, sample_rate, self.sample_rate)

        return stream

    async def __main(self, voicename: str, text: str):
        communicate = edge_tts.Communicate(text, voicename)
        first = True
        async for chunk in communicate.stream():
            if first:
                first = False
            if chunk["type"] == "audio" and self.state == State.RUNNING:
                self.input_stream.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass

###########################################################################################
class VoitsTTS(BaseTTS):
    async def txt_to_audio(self, msg):
        audio_stream = self.gpt_sovits(
            msg,
            self.opt.REF_FILE,
            self.opt.REF_TEXT,
            "zh",
            self.opt.TTS_SERVER,
        )
        await self.stream_tts(audio_stream)

    async def gpt_sovits(self, text, reffile, reftext, language, server_url) -> AsyncIterator[bytes]:
        start = time.perf_counter()
        req = {
            'text': text,
            'text_lang': language,
            'ref_audio_path': reffile,
            'prompt_text': reftext,
            'prompt_lang': language,
            'media_type': 'raw',
            'streaming_mode': True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/tts", json=req) as res:
                end = time.perf_counter()
                print(f"gpt_sovits Time to make POST: {end - start}s")

                if res.status != 200:
                    print("Error:", await res.text())
                    return

                first = True
                async for chunk in res.content.iter_chunked(32000):
                    if first:
                        end = time.perf_counter()
                        print(f"gpt_sovits Time to first chunk: {end - start}s")
                        first = False
                    if chunk and self.state == State.RUNNING:
                        yield chunk

                print("gpt_sovits response.elapsed:", res.headers.get('X-Response-Time'))

    async def stream_tts(self, audio_stream):
        async for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                loop = asyncio.get_event_loop()
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = await loop.run_in_executor(None, resampy.resample, stream, 32000, self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.speaker = None

    async def txt_to_audio(self, msg):
        if not self.speaker:
            self.speaker = await self.get_speaker(self.opt.REF_FILE, self.opt.TTS_SERVER)
        audio_stream = self.xtts(
            msg,
            self.speaker,
            "zh-cn",
            self.opt.TTS_SERVER,
            "20"
        )
        await self.stream_tts(audio_stream)

    async def get_speaker(self, ref_audio, server_url):
        data = aiohttp.FormData()
        data.add_field('wav_file', open(ref_audio, 'rb'), filename='reference.wav')
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/clone_speaker", data=data) as response:
                return await response.json()

    async def xtts(self, text, speaker, language, server_url, stream_chunk_size) -> AsyncIterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{server_url}/tts_stream", json=speaker) as res:
                end = time.perf_counter()
                print(f"xtts Time to make POST: {end - start}s")

                if res.status != 200:
                    print("Error:", await res.text())
                    return

                first = True
                async for chunk in res.content.iter_chunked(960):
                    if first:
                        end = time.perf_counter()
                        print(f"xtts Time to first chunk: {end - start}s")
                        first = False
                    if chunk:
                        yield chunk

                print("xtts response.elapsed:", res.headers.get('X-Response-Time'))

    async def stream_tts(self, audio_stream):
        async for chunk in audio_stream:
            if chunk and len(chunk) > 0:
                loop = asyncio.get_event_loop()
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = await loop.run_in_executor(None, resampy.resample, stream, 24000, self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx + self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk
