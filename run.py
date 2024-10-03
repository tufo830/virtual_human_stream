# server.py
import base64
import time
import json
import os
import re
import numpy as np
import multiprocessing
import argparse
import shutil
import asyncio

from aiohttp import web
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from threading import Thread, Event

from webrtc import HumanPlayer

# Import your LLM model (make sure to adjust the import path)
from llm.LLM import LLM

nerfreals = []
statreals = []

# Initialize the set of PeerConnections
pcs = set()

# Handler for the '/humanecho' WebSocket route
async def humanecho(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('WebSocket connection established for /humanecho')
    sessionid = request.rel_url.query.get('sessionid', 0)
    nerfreal = nerfreals[int(sessionid)]
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                message = msg.data.strip()
                if message:
                    nerfreal.put_msg_txt(message)
                else:
                    await ws.send_str('Input message is empty')
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
    finally:
        print('WebSocket connection closed for /humanecho')
    return ws

# Handler for the '/humanchat' WebSocket route
async def humanchat(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('WebSocket connection established for /humanchat')
    sessionid = request.rel_url.query.get('sessionid', 0)
    nerfreal = nerfreals[int(sessionid)]
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                message = msg.data.strip()
                if message:
                    res = await llm_response(message)
                    nerfreal.put_msg_txt(res)
                else:
                    await ws.send_str('Input message is empty')
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
    finally:
        print('WebSocket connection closed for /humanchat')
    return ws

# Function to get response from the LLM
async def llm_response(message):
    # Initialize your LLM model
    llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, llm.chat, message)
    print(response)
    return response

# Handler for the '/offer' route
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = len(nerfreals)
    for index, value in enumerate(statreals):
        if value == 0:
            sessionid = index
            break
    if sessionid >= len(nerfreals):
        print('Reached max session limit')
        return web.Response(status=500, text='Reached max session limit')
    statreals[sessionid] = 1

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)
            statreals[sessionid] = 0

    player = HumanPlayer(nerfreals[sessionid])
    pc.addTrack(player.audio)
    pc.addTrack(player.video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }
        ),
    )

# Handler for the '/human' route
async def human(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    nerfreal = nerfreals[int(sessionid)]

    if params.get('interrupt'):
        nerfreal.pause_talk()

    if params['type'] == 'echo':
        nerfreal.put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        res = await llm_response(params['text'])
        nerfreal.put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

# Handler for the '/set_audiotype' route
async def set_audiotype(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    nerfreal = nerfreals[int(sessionid)]
    nerfreal.set_curr_state(params['audiotype'], params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": 0, "data": "ok"}),
    )

# Shutdown handler to close peer connections
async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# Function to start the server
def run_server(opt):
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    # Add routes
    app.router.add_get('/humanecho', humanecho)
    app.router.add_get('/humanchat', humanchat)
    app.router.add_post('/offer', offer)
    app.router.add_post('/human', human)
    app.router.add_post('/set_audiotype', set_audiotype)
    app.router.add_static('/', path='web')

    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    # Run the app
    web.run_app(app, host='0.0.0.0', port=opt.listenport)

# Main execution
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    # (Your argument definitions go here)
    # For brevity, I'm skipping the parser definitions
    # Make sure to include all your command-line arguments as per your original code

    opt = parser.parse_args()
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    # Initialize your models based on opt.model
    if opt.model == 'ernerf':
        from ernerf.nerf_triplane.provider import NeRFDataset_Test
        from ernerf.nerf_triplane.utils import *
        from ernerf.nerf_triplane.network import NeRFNetwork
        from nerfreal import NeRFReal

        opt.test = True
        opt.test_train = False
        opt.fp16 = True
        opt.cuda_ray = True
        opt.exp_eye = True
        opt.smooth_eye = True

        if opt.torso_imgs == '':
            opt.torso = True

        seed_everything(opt.seed)
        print(opt)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt)

        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []

        print(model)
        trainer = Trainer(
            'ngp', opt, model, device=device, workspace=opt.workspace,
            criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt
        )

        test_loader = NeRFDataset_Test(opt, device=device).dataloader()
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        for _ in range(opt.max_session):
            nerfreal = NeRFReal(opt, trainer, test_loader)
            nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        print(opt)
        for _ in range(opt.max_session):
            nerfreal = MuseReal(opt)
            nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal
        print(opt)
        for _ in range(opt.max_session):
            nerfreal = LipReal(opt)
            nerfreals.append(nerfreal)

    for _ in range(opt.max_session):
        statreals.append(0)

    if opt.transport == 'rtmp':
        thread_quit = Event()
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

    # Start the server
    run_server(opt)

