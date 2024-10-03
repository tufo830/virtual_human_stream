1.  1. PyTorch3D installation unsuccessful.
Download source code and compile.

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
python setup.py install
```

2.  WebSocket connection error.
Modify python/site-packages/flask_sockets.py

```python
self.url_map.add(Rule(rule, endpoint=f)) change to 
self.url_map.add(Rule(rule, endpoint=f, websocket=True))
```

3. The protobuf version is too high.

```bash
pip uninstall protobuf
pip install protobuf==3.20.1
```

4. The digital human does not blink.
Add the following steps when training the model.

> Obtain AU45 for eyes blinking.\
> Run FeatureExtraction in OpenFace, rename and move the output CSV file to data/\<ID>/au.csv.

Copy au.csv to the data directory of this project.

5. Add a background image to the digital human.

```bash
python app.py --bg_img bc.jpg
```

6. Using a self-trained model results in a dimension mismatch error.
Extract audio features using wav2vec when training the model.

```bash
python main.py data/ --workspace workspace/ -O --iters 100000 --asr_model cpierse/wav2vec2-large-xlsr-53-esperanto
```

7. When streaming with RTMP, the ffmpeg version is incorrect. Feedback from online users indicates that version 4.2.2 is required. I'm not sure which versions are incompatible. The principle is to run ffmpeg and check the printed information for 'libx264'; if it's not present, it definitely won't work.
```
--enable-libx264
```
8. Replace with your own trained model.
```python
.
├── data
│   ├── data_kf.json （Corresponding to transforms_train.json in the training data.）
│   ├── au.csv			
│   ├── pretrained
│   └── └── ngp_kf.pth （Corresponding to the trained model ngp_ep00xx.pth.）

```


