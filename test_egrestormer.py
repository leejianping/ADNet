
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.archs.restormer_arch import Restormer
from basicsr.archs.eg_restormer_arch import EGRestormer


from skimage import img_as_ubyte
from pdb import set_trace as stx
import time

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='./datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/tjy/ljp/qrgan/experiments/egrestormer/models/net_g_336000.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='QRcode', type=str, help='Test Dataset') # ['GoPro', 

args = parser.parse_args()

####### Load yaml #######change_1
#yaml_file = 'options/train/train_restormer.yml'
yaml_file = 'options/train/train_egrestormer_qrdataset.yml'
#yaml_file = 'options/train/train_egrestormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

##########################change_2

#model_restoration = Restormer(**x['network_g'])
model_restoration = EGRestormer(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

##########################change_3 last
#inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input_realblur')
#inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input_test')
inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input_150')
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

inference_times = []
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        
        # ======================================================================
        # NEW: Measure inference time
        # ======================================================================
        torch.cuda.synchronize()  # Ensure all previous operations are finished
        
        start_event.record()
        restored = model_restoration(input_)
        end_event.record()
        
        torch.cuda.synchronize()  # Wait for the inference to complete
        
        # Calculate time in seconds and store it
        current_inference_time = start_event.elapsed_time(end_event) / 1000.0
        inference_times.append(current_inference_time)
        # ======================================================================

        # Unpad images to original dimensions
        restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))

if inference_times:
    total_time = sum(inference_times)
    average_inference_time = total_time / len(inference_times)
    
    print("\n" + "="*40)
    print("Inference Time Summary")
    print("="*40)
    print(f"Total images processed: {len(inference_times)}")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average inference time per image: {average_inference_time:.4f} seconds")
    print("="*40)
else:
    print("No images were processed.")
