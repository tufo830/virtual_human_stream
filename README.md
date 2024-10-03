Real time interactive streaming digital human， realize audio video synchronous dialogue. It can basically achieve commercial effects.  

## Features
1. Supports various digital human models: ernerf, musetalk, wav2lip  
2. Supports voice cloning  
3. Supports digital humans being interrupted while speaking  
4. Supports full-body video stitching  
5. Supports RTMP and WebRTC  
6. Supports video editing: plays custom videos when not speaking  

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#If only using the musetalk or wav2lip models, there is no need to install the following libraries.
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install --upgrade "protobuf<=3.20.1"
```
Installation Common Questions FAQ[FAQ](/assets/faq.md)  

## 2. Quick Start
By default, the ernerf model is used, and WebRTC is used to stream to SRS. 
### 2.1 run srs
```
export CANDIDATE='<Server network ip>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

### 2.2 Start the digital human：

```python
python run.py
```

Open http://serverip:8010/rtcpushapi.html in a browser, enter any text in the textbox, and submit. The digital human will broadcast the entered text. 
Note: The server needs to open the port tcp:8000,8010,1985; udp:8000

## 3. More Usage
### 3.1 Using LLM model for digital human dialogue

Currently,where the LLM model supports ChatGPT, Qwen, and GeminiPro. You need to enter your own api_key in run.py.

Open with browser:http://serverip:8010/rtcpushchat.html

### 3.2 voice cloning
You can choose from the following two services, with gpt-sovits recommended
#### 3.2.1 gpt-sovits
Service deployment reference[gpt-sovits](/tts/README.md)  
run
```
python run.py --tts gpt-sovits --TTS_SERVER http://127.0.0.1:9880 --REF_FILE data/ref.wav --REF_TEXT xxx
```
The REF_TEXT is the voice content in REF_FILE, and the duration should not be too long.

#### 3.2.2 xtts
Run the xtts service
```
docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 9000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest
```
Then run it, where ref.wav is the voice file that needs to be cloned
```
python run.py --tts xtts --REF_FILE data/ref.wav --TTS_SERVER http://localhost:9000
```

### 3.3 Use HuBERT for audio features
If HuBERT is used to extract audio features during model training, use the following command to start the digital human
```
python run.py --asr_model facebook/hubert-large-ls960-ft 
```

### 3.4 Set background image
```
python run.py --bg_img bc.jpg 
```

### 3.5 Full-body video stitching
#### 3.5.1 Cut videos for training
```
ffmpeg -i fullbody.mp4 -vf crop="400:400:100:5" train.mp4 
```
Train the model with train.mp4
#### 3.5.2 Extract full-body images
```
ffmpeg -i fullbody.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/fullbody/img/%d.jpg
```
#### 3.5.2 Start the digital human
```
python run.py --fullbody --fullbody_img data/fullbody/img --fullbody_offset_x 100 --fullbody_offset_y 5 --fullbody_width 580 --fullbody_height 1080 --W 400 --H 400
```
- --fullbody_width, --fullbody_height are the width and height of the full-body video
- --W, --H are the width and height of the training video  
- For the third step of ernerf training, the torso, if not trained well, seams may runear at the joints. You can add --torso_imgs data/xxx/torso_imgs to the command above. For the torso, instead of model inference, use the torso images directly from the training dataset. This method may result in some artificial marks at the neck and head.

### 3.6 Replace with a custom video when not speaking
- Extract images from a custom video
```
ffmpeg -i silence.mp4 -vf fps=25 -qmin 1 -q:v 1 -start_number 0 data/customvideo/img/%d.png
```
- Run the digital human
```
python run.py --customvideo --customvideo_img data/customvideo/img --customvideo_imgnum 100
```

### 3.7 webrtc p2p
This mode does not require srs
```
python run.py --transport webrtc
```
The server needs to open the ports tcp:8010 and udp:50000~60000.
Open http://serverip:8010/webrtcapi.html in a browser.

### 3.8 rtmp push to srs
- Install the rtmpstream library

- run srs
```
docker run --rm -it -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5
```
- Run the digital human
```python
python run.py --transport rtmp --push_url 'rtmp://localhost/live/livestream'
```
Open http://serverip:8010/echoapi.html in a browser.

### 3.9 Use the musetalk model
RTMP push is not supported for now
- Install the dependency libraries
```bash
conda install ffmpeg
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"
```
- Download the model
Download the model required to run MuseTalk from the following link: https://caiyun.139.com/m/i?2eAjs2nXXnRgr Access code: qdg2 
After extraction, copy the files under the models directory to the models directory of this project. 
Download the digital human model from this link: https://caiyun.139.com/m/i?2eAjs8optksop Access code: 3mkt, and after extraction, copy the entire folder to the data/avatars directory of this project.
- run  
python run.py --model musetalk --transport webrtc  
Open http://serverip:8010/webrtcapi.html in a browser
You can set --batch_size to improve GPU utilization, and set --avatar_id to run different digital humans
#### Replace with your own digital human
```bash
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
Modify configs/inference/realtime.yaml and set preparation to True
python -m scripts.realtime_inference --inference_config configs/inference/realtime.yaml
After running, copy the files from results/avatars to the data/avatars directory of this project.
Method 2:
Execute
cd musetalk 
python musetalk.py --avatar_id 4  --file yourpath
Supports video and image generation, which will be automatically generated in the avatars directory under data
```

### 3.10 Use the wav2lip model
rtmp push is not supported for now
- Download the model  
Download the model required to run wav2lip from the following link: https://pan.baidu.com/s/1yOsQ06-RIDTJd3HFCw4wtA Password: ltua
Copy s3fd.pth to wav2lip/face_detection/detection/sfd/s3fd.pth in this project, and copy wav2lip.pth to the models directory in this project 
The digital human model file wav2lip_avatar1.tar.gz, after extraction, should be copied as a whole folder to the data/avatars directory in this project
- run  
python run.py --transport webrtc --model wav2lip --avatar_id wav2lip_avatar1  
Open http://serverip:8010/webrtcapi.html in a browser
You can set --batch_size to improve GPU utilization and set --avatar_id to run different digital humans
#### Replace with your own digital human
```bash
cd wav2lip
python genavatar.py --video_path xxx.mp4
After running, copy the files from results/avatars to the data/avatars directory in this project
```

## 4. The digital human model files
 can be replaced with models you have trained yourself(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── data_kf.json
│   ├── au.csv			
│   ├── pretrained
│   └── └── ngp_kf.pth

```

## 5. Performance analysis and optimization strategies
# Performance Analysis and Optimization Strategies

## 1. Frame Rate Enhancement

### Current Performance
- **GPU**: Tesla T4
- **Frame Rate**: Approximately 18 FPS. Disabling audio and video encoding and streaming increases frame rate to about 20 FPS.

### Hardware Upgrade
- **Upgraded GPU**: RTX 4090
- **Achieved Frame Rate**: Over 40 FPS

### Optimization Strategy
- **Approach**: Implement a new thread dedicated to audio and video encoding and streaming.
- **Benefits**: Reduces bottlenecks associated with single-threaded execution, enhancing frame rate.

## 2. Overall Latency

### Total System Latency
- **Duration**: Around 3 seconds

#### a. TTS Latency (1.7 seconds)
- **Issue**: The edgetts component processes the entire sentence before outputting, causing significant delay.
- **Optimization**: Transition to a streaming input method for TTS, allowing real-time text processing as it is received.

#### b. Wav2Vec Latency (0.4 seconds)
- **Issue**: Requires buffering 18 frames of audio before computation.
- **Optimization**: Explore reducing buffer size or optimizing the model to process fewer frames without losing accuracy.

#### c. SRS Forwarding Latency
- **Issue**: Default buffering settings of the Simple Realtime Server (SRS) introduce latency.
- **Optimization**: Adjust the SRS server settings to minimize buffering delays, such as the `gop_cache` or `queue_length`.


## 6. TODO
 1.Implemented ChatGPT for interactive dialogues with digital humans  
 2.Integrated voice cloning technology  
 3.Enabled video replacement for digital humans when muted  
 4.Implemented MuseTalk feature  
 5.Added Wav2Lip synchronization  
 6.Pending SyncTalk implementation  
 7.Enhance real-time interaction capabilities  
 8.Develop multi-language support for global accessibility  
 9.Incorporate advanced facial expression recognition  
 10.Introduce adaptive learning algorithms for better user personalization


