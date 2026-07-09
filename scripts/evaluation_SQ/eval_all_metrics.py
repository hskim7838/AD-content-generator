import os
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2

# 경로 및 파일 설정
target_dir = 'output_demo/output_image/extra_score_data'
output_file = 'fid_report.md'

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

# 이미지 로드
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
all_imgs = [cv2.imread(os.path.join(target_dir, f)) for f in filenames]
all_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_imgs]

# 기준 이미지 설정
img_org = all_imgs_rgb[0]
org_name = filenames[0]

# 마크다운 작성
md_content = f"# 📊 생성 이미지 종합 평가 리포트\n\n"
md_content += f"| 미리보기 | 파일명 | PSNR (dB) | SSIM | FID 영향도 |\n"
md_content += f"|:---:|---|:---:|:---:|:---:|\n"

gen_images = all_imgs_rgb[1:]
gen_names = filenames[1:]

# 각 지표 계산 및 표 작성
for i, (name, img_gen) in enumerate(zip(gen_names, gen_images)):
    # 크기 정규화
    img_gen_res = cv2.resize(img_gen, (img_org.shape[1], img_org.shape[0]))
    
    # PSNR/SSIM
    p_score = calculate_psnr(img_org, img_gen_res)
    s_score = calculate_ssim(img_org, img_gen_res)
    
    # FID 영향도 (이전 로직 재활용)
    # 생략된 코드: 실제로는 개별 FID 계산 로직을 여기 배치하여 추가
    
    img_tag = f"<img src='./{target_dir}/{name}' width='100'>"
    md_content += f"| {img_tag} | `{name}` | {p_score:.2f} | {s_score:.4f} | 계산중... |\n"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_content)
