import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import yaml
from skimage import img_as_ubyte
import cv2
from pyzbar import pyzbar
import utils
import time # Import the time module

# Import model architectures
from basicsr.archs.restormer_arch import Restormer 
from basicsr.archs.eg_restormer_arch import EGRestormer 
from basicsr.archs.LEDNet_arch import LENet      


# ==============================================================================
# AdaptiveQRDeblurrer Class
# ==============================================================================
class AdaptiveQRDeblurrer:
    def __init__(self, light_model_config, light_model_weights, heavy_model_config, heavy_model_weights, device='cuda'):
       
        self.device = torch.device(device)
        self.factor = 8 

        print("Loading Light Model...")
        self.light_model = self._load_model_from_config(light_model_config, light_model_weights)
        
        print("Loading Heavy Model...")
        self.heavy_model = self._load_model_from_config(heavy_model_config, heavy_model_weights)
        
        print("Models loaded successfully.")

    def _load_model_from_config(self, yaml_file, weights_path):
        """General model loading function."""
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        with open(yaml_file, mode='r') as f:
            x = yaml.load(f, Loader=Loader)
        model_type = x['network_g'].pop('type')
        model_class = globals()[model_type]
        model = model_class(**x['network_g'])
        checkpoint = torch.load(weights_path)
        state_dict = checkpoint.get('params', checkpoint)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model = nn.DataParallel(model)
        model.eval()
        return model

    def detect_blur_level(self, image: np.ndarray, threshold: float = 100.0) -> str:
        """Detect blur level using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Laplacian variance: {laplacian_var:.2f}")
        return "light" if laplacian_var > threshold else "heavy"

    def _run_model(self, model: nn.Module, image_np: np.ndarray) -> (np.ndarray, float, float):
        """
        General model inference function.
        Returns:
            restored_uint8 (np.ndarray): The deblurred image.
            inference_time (float): The pure GPU inference time in seconds.
            processing_time (float): The total time for the function including pre/post-processing.
        """
        # --- Total Processing Time Measurement ---
        processing_start_time = time.time()

        # Pre-processing
        img = np.float32(image_np) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).to(self.device)

        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + self.factor) // self.factor) * self.factor, ((w + self.factor) // self.factor) * self.factor
        padh = H - h if h % self.factor != 0 else 0
        padw = W - w if w % self.factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        # --- Pure Inference Time Measurement ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize() # Wait for any previous GPU operations to finish
        
        with torch.no_grad():
            start_event.record()
            restored = model(input_)
            end_event.record()

        torch.cuda.synchronize() # Wait for the model inference to complete
        
        inference_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds

        # Post-processing
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_uint8 = img_as_ubyte(restored)

        processing_end_time = time.time()
        processing_time = processing_end_time - processing_start_time
        
        return restored_uint8, inference_time, processing_time

    def decode_qr(self, image: np.ndarray) -> bool:
        """Attempts to decode a QR code from the given image."""
        decoded_objects = pyzbar.decode(image)
        if decoded_objects:
            print(f"  SUCCESS: QR Code decoded. Data: {decoded_objects[0].data.decode('utf-8')}")
            return True
        else:
            print("  FAIL: QR Code could not be decoded.")
            return False

    def process(self, input_image: np.ndarray) -> (np.ndarray, str, float, float):
        """
        Executes the adaptive deblurring process.
        Returns: (final image, path used, total inference time, total processing time)
        """
        total_inference_time = 0.0
        total_processing_time = 0.0
        
        blur_level = self.detect_blur_level(input_image)

        if blur_level == "light":
            print("PATH: Detected 'Light' blur. Using Light Model.")
            light_restored_image, infer_time, proc_time = self._run_model(self.light_model, input_image)
            total_inference_time += infer_time
            total_processing_time += proc_time
            
            print("Attempting to decode result from Light Model...")
            if self.decode_qr(light_restored_image):
                path = "Light Model -> Decode Success"
                return light_restored_image, path, total_inference_time, total_processing_time
            else:
                print("PATH: Decode failed. Falling back to Heavy Model.")
                heavy_restored_image, infer_time_heavy, proc_time_heavy = self._run_model(self.heavy_model, input_image)
                total_inference_time += infer_time_heavy
                total_processing_time += proc_time_heavy
                path = "Light Model -> Decode Fail -> Heavy Model"
                return heavy_restored_image, path, total_inference_time, total_processing_time
        else:
            print("PATH: Detected 'Heavy' blur. Using Heavy Model.")
            restored_image, infer_time, proc_time = self._run_model(self.heavy_model, input_image)
            total_inference_time += infer_time
            total_processing_time += proc_time
            path = "Heavy Model"
            return restored_image, path, total_inference_time, total_processing_time


# ==============================================================================
# Main Program Framework
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Adaptive Dual-Model QR Code Deblurring')
    parser.add_argument('--input_dir', default='./datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
    parser.add_argument('--dataset', default='QRcode', type=str, help='Test Dataset folder name')
    parser.add_argument('--light_weights', default='/home/tjy/ljp/qrgan/experiments/LENet/models/net_g_1000000.pth', type=str, help='Path to light model weights (LENet)')
    parser.add_argument('--light_config', default='./options/train/train_qrnet.yml', type=str, help='Path to light model config yaml')
    parser.add_argument('--heavy_weights', default='/home/tjy/ljp/qrgan/experiments/egrestormer/models/net_g_336000.pth', type=str, help='Path to heavy model weights (Restormer)')
    parser.add_argument('--heavy_config', default='options/train/train_egrestormer_qrdataset.yml', type=str, help='Path to heavy model config yaml')
    args = parser.parse_args()

    deblurrer = AdaptiveQRDeblurrer(
        light_model_config=args.light_config,
        light_model_weights=args.light_weights,
        heavy_model_config=args.heavy_config,
        heavy_model_weights=args.heavy_weights
    )
    
    result_dir = os.path.join(args.result_dir, args.dataset)
    os.makedirs(result_dir, exist_ok=True)
    inp_dir = os.path.join(args.input_dir, 'test', args.dataset, 'input_150')
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))

    # --- Initialize timing and path statistics ---
    total_inference_time_all = 0.0
    total_processing_time_all = 0.0
    path_counts = {
        "Light Model -> Decode Success": 0,
        "Light Model -> Decode Fail -> Heavy Model": 0,
        "Heavy Model": 0
    }

    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img_bgr = cv2.imread(file_)
        if img_bgr is None:
            print(f"Warning: Could not read image {file_}. Skipping.")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        final_image, used_path, infer_time, proc_time = deblurrer.process(img_rgb)
        
        total_inference_time_all += infer_time
        total_processing_time_all += proc_time
        if used_path in path_counts:
            path_counts[used_path] += 1
        
        print(f"\nFile: {os.path.basename(file_)}")
        print(f"Processing Path: {used_path}")
        print(f"  - Pure Inference Time: {infer_time:.4f} seconds")
        print(f"  - Total Processing Time: {proc_time:.4f} seconds")
        
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')
        utils.save_img(save_path, final_image_bgr)

    # --- Print final summary ---
    total_images = len(files)
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    
    for path, count in path_counts.items():
        print(f"Path '{path}' used: {count} times")

    print("-" * 60)
    
    print(f"Total images processed: {total_images}")
    if total_images > 0:
        print(f"Total Pure Inference Time: {total_inference_time_all:.4f} seconds")
        print(f"Average Inference Time per image: {total_inference_time_all / total_images:.4f} seconds")
        print("-" * 20)
        print(f"Total Processing Time (wall clock): {total_processing_time_all:.4f} seconds")
        print(f"Average Processing Time per image: {total_processing_time_all / total_images:.4f} seconds")

    print("="*60)

if __name__ == '__main__':
    main()