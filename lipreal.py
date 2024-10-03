import math
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import pickle
import copy

import asyncio
from av import AudioFrame, VideoFrame
from wav2lip.models import Wav2Lip
from basereal import BaseReal

from ttsreal import EdgeTTS, VoitsTTS, XTTS
from lipasr import LipASR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

def read_imgs(img_list):
    frames = []
    print('Reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1

class InferenceWorker:
    def __init__(self, render_event, batch_size, face_imgs_path, audio_feat_queue, audio_out_queue):
        self.render_event = render_event
        self.batch_size = batch_size
        self.face_imgs_path = face_imgs_path
        self.audio_feat_queue = audio_feat_queue
        self.audio_out_queue = audio_out_queue
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self.process = mp.Process(target=self.run)
        self.process.start()

    def run(self):
        model = load_model("./models/wav2lip.pth")
        input_face_list = sorted(glob.glob(os.path.join(self.face_imgs_path, '*.[jpJP][pnPN]*[gG]')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        face_list_cycle = read_imgs(input_face_list)
        length = len(face_list_cycle)
        index = 0
        count = 0
        counttime = 0
        print('Start inference worker')
        while True:
            if self.render_event.is_set():
                starttime = time.perf_counter()
                try:
                    mel_batch = self.audio_feat_queue.get(timeout=1)
                except queue.Empty:
                    continue

                is_all_silence = True
                audio_frames = []
                for _ in range(self.batch_size * 2):
                    frame, type = self.audio_out_queue.get()
                    audio_frames.append((frame, type))
                    if type == 0:
                        is_all_silence = False

                if is_all_silence:
                    for i in range(self.batch_size):
                        self.res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        index += 1
                else:
                    t = time.perf_counter()
                    img_batch = []
                    for i in range(self.batch_size):
                        idx = __mirror_index(length, index + i)
                        face = face_list_cycle[idx]
                        img_batch.append(face)
                    img_batch = np.asarray(img_batch)
                    mel_batch = np.asarray(mel_batch)

                    img_masked = img_batch.copy()
                    img_masked[:, face.shape[0]//2:, :] = 0

                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                    with torch.no_grad():
                        pred = model(mel_batch, img_batch)
                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                    counttime += (time.perf_counter() - t)
                    count += self.batch_size
                    if count >= 100:
                        print(f"------actual avg infer fps: {count / counttime:.4f}")
                        count = 0
                        counttime = 0
                    for i, res_frame in enumerate(pred):
                        self.res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                        index += 1
            else:
                time.sleep(1)
        print('Inference worker stopped')

    def get_result(self):
        return self.res_frame_queue

    def stop(self):
        self.process.terminate()
        self.process.join()

class LipReal(BaseReal):
    def __init__(self, opt):
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps  # 20 ms per frame

        self.avatar_id = opt.avatar_id
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.face_imgs_path = f"{self.avatar_path}/face_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.batch_size = opt.batch_size
        self.idx = 0

        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self.__load_avatar()

        self.asr = LipASR(opt, self)
        self.asr.warm_up()
        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt, self)
        elif opt.tts == "gpt-sovits":
            self.tts = VoitsTTS(opt, self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt, self)

        self.render_event = mp.Event()
        self.inference_worker = InferenceWorker(self.render_event, self.batch_size, self.face_imgs_path, self.asr.feat_queue, self.asr.output_queue)

    def __load_avatar(self):
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()

    async def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = await asyncio.get_event_loop().run_in_executor(None, self.inference_worker.get_result().get, True, 1)
            except queue.Empty:
                continue
            if all(af[1] != 0 for af in audio_frames):  # All silence
                combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
                y1, y2, x1, x2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                combine_frame[y1:y2, x1:x2] = res_frame

            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            await video_track._queue.put(new_frame)

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                await audio_track._queue.put(new_frame)
        print('LipReal process_frames coroutine stopped')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)
        asyncio.ensure_future(self.process_frames(quit_event, loop, audio_track, video_track))

        self.render_event.set()  # Start inference process render
        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()
            if video_track._queue.qsize() >= 5:
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        self.render_event.clear()  # End inference process render
        print('LipReal render loop stopped')
