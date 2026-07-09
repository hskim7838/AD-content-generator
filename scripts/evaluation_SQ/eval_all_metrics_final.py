import os
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2

target_dir = '/home/ai3/AD-content/test_data'
output_file = 'fid_report.md'

# 기본 도구 함수들
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1, mu2 = cv2.GaussianBlur(img1, (11, 11), 1.5), cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    return np.mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

# 이미지 로드
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
all_imgs = [cv2.resize(cv2.imread(os.path.join(target_dir, f)), (299, 299)) for f in filenames]
all_imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_imgs]

org_img = torch.tensor(all_imgs_rgb[0]).permute(2,0,1).unsqueeze(0)
gen_imgs_tensor = torch.stack([torch.tensor(img).permute(2,0,1) for img in all_imgs_rgb[1:]])

# 전체 Baseline FID 계산
fid_metric = FrechetInceptionDistance(feature=64, reset_real_features=True)
fid_metric.update(torch.stack([org_img.squeeze(0)] * len(gen_imgs_tensor)), real=True)
fid_metric.update(gen_imgs_tensor, real=False)
baseline_fid = float(fid_metric.compute())

# 마크다운 작성
md_content = f"# 📊 생성 이미지 종합 평가 리포트\n\n| 미리보기 | 파일명 | PSNR (dB) | SSIM | FID 영향도 |\n|:---:|---|:---:|:---:|:---:|\n"

for i, (name, img_gen) in enumerate(zip(filenames[1:], all_imgs_rgb[1:])):
    p_score = calculate_psnr(all_imgs_rgb[0], img_gen)
    s_score = calculate_ssim(all_imgs_rgb[0], img_gen)
    
    # 해당 이미지 제외 FID
    sub_gen = torch.cat([gen_imgs_tensor[:i], gen_imgs_tensor[i+1:]])
    fid_sub = FrechetInceptionDistance(feature=64, reset_real_features=True)
    fid_sub.update(torch.stack([org_img.squeeze(0)] * len(sub_gen)), real=True)
    fid_sub.update(sub_gen, real=False)
    diff = float(fid_sub.compute()) - baseline_fid
    
    img_tag = f"<img src='./{target_dir}/{name}' width='100'>"
    md_content += f"| {img_tag} | `{name}` | {p_score:.2f} | {s_score:.4f} | {diff:+.4f} |\n"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_content)
print("✅ 종합 리포트 생성 완료! fid_report.md를 확인하세요.")
