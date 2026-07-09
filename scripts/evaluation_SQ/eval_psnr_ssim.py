import os
import cv2
import numpy as np

# 모든 이미지가 들어있는 실제 경로로 지정
target_dir = '/home/ai3/AD-content/CAIG/output_demo/output_image/extra_score_data'

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
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

# 폴더 내 순수 이미지 파일만 가져오기
all_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not all_files:
    print(f"❌ 폴더에 이미지 파일이 없습니다: {target_dir}")
    exit()

# 첫 번째 이미지를 기준(원본 대조군)으로 설정
org_name = all_files[0]
org_path = os.path.join(target_dir, org_name)
img_org = cv2.imread(org_path)

psnr_list = []
ssim_list = []

print(f"\n🎯 [기준 이미지]: {org_name}")
print("🔍 --- [생성 이미지별 PSNR / SSIM 측정 결과] ---")
print(f"{'생성 파일명':<45} | {'PSNR (dB)':<10} | {'SSIM':<6}")
print("-" * 75)

# 기준 이미지를 제외한 나머지 이미지들과 비교
for gen_name in all_files[1:]:
    gen_path = os.path.join(target_dir, gen_name)
    img_gen = cv2.imread(gen_path)
    
    if img_gen is None:
        continue
        
    # 크기 동기화
    img_gen_resized = cv2.resize(img_gen, (img_org.shape[1], img_org.shape[0]))
    
    p_score = calculate_psnr(img_org, img_gen_resized)
    s_score = calculate_ssim(img_org, img_gen_resized)
    
    psnr_list.append(p_score)
    ssim_list.append(s_score)
    
    print(f"{gen_name:<45} | {p_score:<10.4f} | {s_score:<6.4f}")

print("-" * 75)
if psnr_list:
    print(f"📊 [전체 {len(psnr_list)}장 평가 결과]")
    print(f"✅ 평균 PSNR Score: {np.mean(psnr_list):.4f} dB")
    print(f"✅ 평균 SSIM Score: {np.mean(ssim_list):.4f}")