import os
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2

target_dir = 'output_demo/output_image/extra_score_data'

def load_images_from_dir(directory):
    images = []
    if not os.path.exists(directory):
        print(f"❌ 폴더 경로가 존재하지 않습니다: {os.path.abspath(directory)}")
        return images
        
    all_files = os.listdir(directory)
    filenames = sorted([f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for f in filenames:
        full_path = os.path.join(directory, f)
        img = cv2.imread(full_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (299, 299))
            img = np.transpose(img, (2, 0, 1))
            images.append((f, torch.tensor(img, dtype=torch.uint8)))
    return images

print("🔄 이미지 로딩 중...")
all_images = load_images_from_dir(target_dir)

if not all_images:
    print(f"❌ 폴더에 이미지 파일이 없습니다. 탐색 경로: {os.path.abspath(target_dir)}")
    exit()

org_name, img_org_tensor = all_images[0]
gen_images = all_images[1:]

if not gen_images:
    print("❌ 비교할 생성 이미지가 부족합니다. (최소 2장 이상 필요)")
    exit()

real_imgs = torch.stack([img_org_tensor] * len(gen_images))
gen_imgs_tensor = torch.stack([img for _, img in gen_images])

fid_metric = FrechetInceptionDistance(feature=64, reset_real_features=True)
fid_metric.update(real_imgs, real=True)
fid_metric.update(gen_imgs_tensor, real=False)
baseline_fid = float(fid_metric.compute())

print(f"\n🎯 [기준 대조 이미지]: {org_name}")
print(f"📊 기준 전체 FID Score: {baseline_fid:.4f}")
print("🔍 --- [특정 생성 이미지를 제외했을 때의 FID 변화 (영향도 분석)] ---")
print(f"{'제외된 생성 파일명':<45} | {'제외 후 FID':<12} | {'영향도 (변화량)'}")
print("-" * 85)

for i, (filename, _) in enumerate(gen_images):
    sub_gen_imgs = torch.stack([img for j, (_, img) in enumerate(gen_images) if j != i])
    sub_real_imgs = torch.stack([img_org_tensor] * len(sub_gen_imgs))
    
    fid_tmp = FrechetInceptionDistance(feature=64, reset_real_features=True)
    fid_tmp.update(sub_real_imgs, real=True)
    fid_tmp.update(sub_gen_imgs, real=False)
    current_fid = float(fid_tmp.compute())
    
    diff = current_fid - baseline_fid
    print(f"{filename:<45} | {current_fid:<12.4f} | {diff:+.4f}")
