from natsort import natsorted
import os
from tqdm import tqdm
import soundfile as sf
import librosa
import torch
from transformers import WavLMModel, AutoFeatureExtractor
import cv2
import numpy as np
from PIL import Image

# Get all files in a folder in a sorted order.
def get_files(dir):
    files = natsorted(os.listdir(dir))
    return [os.path.join(dir, i) for i in files]

# fid: file id. 
def get_fid(file):
    return os.path.basename(file).split(".")[0]

# Make a directory. 
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        

save_dir = "/data/lie_detection/clips_umich/viT"
mkdir(save_dir)

# Save images.
temp_dir = "/data/lie_detection/clips_umich/temp"
mkdir(temp_dir)

import clip
device = torch.device('cuda:7')
model, preprocess = clip.load("ViT-B/32", device=device)

for file in get_files("/data/lie_detection/clips_umich/video"):
    fid = get_fid(file)
    save_path = os.path.join(save_dir, f"{fid}.npy")
    if os.path.exists(save_path):
        continue
    
    cap = cv2.VideoCapture(file)
    frames = []
    count = 0
    progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(temp_dir, f"{count}.png"), frame)     # save frame as PnG file
        count += 1
        progress_bar.update(1)
    cap.release()
    
    images = [Image.open(i).convert('RGB') for i in get_files(temp_dir)]
    frames = torch.stack([preprocess(i).to(device) for i in images])
    with torch.no_grad():
        output = model.encode_image(frames)
    np.save(save_path, output.cpu().numpy())
    
    # Clean up
    for i in get_files(temp_dir):
        os.remove(i)
    assert len(get_files(temp_dir)) == 0