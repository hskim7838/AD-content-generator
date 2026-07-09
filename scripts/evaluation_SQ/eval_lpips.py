import os
import cv2
import torch
import lpips
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision import transforms

# 설정 # 이미지들이 위치한 폴더
DATA_DIR = '/home/ai3/AD-content/test_data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LPIPS 모델 로드 (vgg 방식)
loss_fn = lpips.LPIPS(net='vgg').to(device)

def calculate_metrics(img1_path, img2_path):
    # 이미지 로드 및 전처리
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    t1 = transform(img1).unsqueeze(0).to(device) * 2 - 1  # LPIPS는 -1~1 범위
    t2 = transform(img2).unsqueeze(0).to(device) * 2 - 1
    
    # PSNR, SSIM을 위한 넘파이 변환 (0~255)
    i1 = np.array(img1.resize((256, 256)))
    i2 = np.array(img2.resize((256, 256)))
    
    psnr = cv2.PSNR(i1, i2)
    ssim_val = ssim(i1, i2, channel_axis=2, data_range=255)
    lpips_val = loss_fn(t1, t2).item()
    
    return psnr, ssim_val, lpips_val

# 평가 수행
results = []
# (가정: 폴더 내에 원본과 생성본이 쌍으로 존재한다고 가정)
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])

for i in range(0, len(files), 2):
    name = files[i]
    p, s, l = calculate_metrics(os.path.join(DATA_DIR, files[i]), os.path.join(DATA_DIR, files[i+1]))
    results.append({'File': name, 'PSNR': p, 'SSIM': s, 'LPIPS': l})

# 결과 저장
df = pd.DataFrame(results)
df.to_csv('evaluation_report.csv', index=False)
print("평가 완료! evaluation_report.csv가 생성되었습니다.")
