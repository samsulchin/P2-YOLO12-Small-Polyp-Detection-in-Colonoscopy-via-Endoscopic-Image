#!/usr/bin/env python3
import time
import torch
import cv2
import numpy as np
import pywt
from ultralytics import YOLO

# --- ZERO-DCE IMPLEMENTATION (Matches the DCENet class in 05_preprocess_pipeline.py) ---
import torch.nn as nn
class DCENet(nn.Module):
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        self.relu    = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, padding=1)   
        self.e_conv6 = nn.Conv2d(64, 32, 3, padding=1)   
        self.e_conv7 = nn.Conv2d(64, 3*num_iterations, 3, padding=1)  
        self.tanh    = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = self.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        return x_r

    def enhance(self, x):
        x_r = self.forward(x)
        enhanced = x
        for i in range(self.num_iterations):
            alpha = x_r[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)
        return torch.clamp(enhanced, 0, 1)

def apply_dwt(img_bgr):
    channels = cv2.split(img_bgr)
    denoised = []
    for ch in channels:
        cA, (cH, cV, cD) = pywt.dwt2(ch.astype(np.float32), 'haar')
        # Lightweight DWT
        rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        denoised.append(np.clip(rec, 0, 255).astype(np.uint8))
    return cv2.merge(denoised)

def apply_sai(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    mask_d = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    return cv2.inpaint(img_bgr, mask_d, 3, cv2.INPAINT_TELEA)

def main():
    print("="*60)
    print("END-TO-END PIPELINE BENCHMARK (Batch=1)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Models
    print("[INFO] Loading models...")
    zerodce = DCENet().to(device).eval()
    
    # IMPORTANT: Replace 'best.pt' with the path to your trained P2-YOLO12 Full weights
    model_path = "best.pt" 
    try:
        yolo_model = YOLO(model_path) 
    except Exception as e:
        print(f"[WARNING] Could not load YOLO model from {model_path}. Using base yolo12s.pt for benchmark.")
        yolo_model = YOLO("yolo12s.pt")
    
    # 2. Dummy Image 640x640
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_tensor = torch.randn(1, 3, 640, 640).to(device)
    
    # 3. WARM-UP (As required for rigorous hardware profiling)
    print("[INFO] Warming up (50 iterations)...")
    for _ in range(50):
        _ = zerodce(img_tensor)
        _ = apply_sai(img)
        _ = apply_dwt(img)
        _ = yolo_model(img, verbose=False)
        
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 4. MEASUREMENT LOOP
    print("[INFO] Measuring latency (100 iterations)...")
    iterations = 100
    
    t_dce, t_sai, t_dwt, t_yolo = 0, 0, 0, 0
    
    for _ in range(iterations):
        # A. Zero-DCE (GPU)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = zerodce(img_tensor)
        torch.cuda.synchronize()
        t_dce += (time.perf_counter() - t0)
        
        # B. SAI (CPU)
        t1 = time.perf_counter()
        img_sai = apply_sai(img)
        t_sai += (time.perf_counter() - t1)
        
        # C. DWT (CPU)
        t2 = time.perf_counter()
        _ = apply_dwt(img_sai)
        t_dwt += (time.perf_counter() - t2)
        
        # D. YOLO (GPU)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        _ = yolo_model(img, verbose=False)
        torch.cuda.synchronize()
        t_yolo += (time.perf_counter() - t3)

    # 5. CALCULATE AVERAGES
    avg_dce = (t_dce / iterations) * 1000
    avg_sai = (t_sai / iterations) * 1000
    avg_dwt = (t_dwt / iterations) * 1000
    avg_yolo = (t_yolo / iterations) * 1000
    
    total_latency = avg_dce + avg_sai + avg_dwt + avg_yolo
    total_fps = 1000 / total_latency
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Zero-DCE Latency : {avg_dce:.2f} ms (GPU)")
    print(f"SAI Latency      : {avg_sai:.2f} ms (CPU)")
    print(f"DWT Latency      : {avg_dwt:.2f} ms (CPU)")
    print(f"Detector Latency : {avg_yolo:.2f} ms (GPU)")
    print("-" * 30)
    print(f"Total Pipeline   : {total_latency:.2f} ms")
    print(f"End-to-End FPS   : {total_fps:.1f} FPS")
    print(f"Peak GPU Memory  : {peak_mem:.2f} MB")
    print("============================================================")

if __name__ == "__main__":
    main()
