import os
import torch
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2

target_dir = '/home/ai3/AD-content/test_data'
output_file = 'fid_report.md' # 결과를 저장할 파일명

def load_images_from_dir(directory):
    images = []
    if not os.path.exists(directory):
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

print("🔄 이미지 로딩 및 FID 점수 계산 중... (시간이 조금 걸릴 수 있습니다)")
all_images = load_images_from_dir(target_dir)

if not all_images:
    print(f"❌ 이미지를 찾을 수 없습니다.")
    exit()

org_name, img_org_tensor = all_images[0]
gen_images = all_images[1:]

real_imgs = torch.stack([img_org_tensor] * len(gen_images))
gen_imgs_tensor = torch.stack([img for _, img in gen_images])

fid_metric = FrechetInceptionDistance(feature=64, reset_real_features=True)
fid_metric.update(real_imgs, real=True)
fid_metric.update(gen_imgs_tensor, real=False)
baseline_fid = float(fid_metric.compute())

# 마크다운 리포트 작성 시작
md_content = f"# 📊 FID Score 시각화 리포트\n\n"
md_content += f"- **기준 원본 이미지:** `{org_name}`\n"
md_content += f"- **기준 전체 FID Score:** `{baseline_fid:.4f}`\n\n"
md_content += f"<img src='{target_dir}/{org_name}' width='250'>\n\n"
md_content += "---\n\n"

md_content += "### 🔍 특정 생성 이미지를 제외했을 때의 FID 변화 (영향도 분석)\n\n"
md_content += "| 미리보기 (생성 이미지) | 제외된 생성 파일명 | 제외 후 FID | 영향도 (변화량) |\n"
md_content += "|:---:|---|:---:|:---:|\n"

for i, (filename, _) in enumerate(gen_images):
    sub_gen_imgs = torch.stack([img for j, (_, img) in enumerate(gen_images) if j != i])
    sub_real_imgs = torch.stack([img_org_tensor] * len(sub_gen_imgs))
    
    fid_tmp = FrechetInceptionDistance(feature=64, reset_real_features=True)
    fid_tmp.update(sub_real_imgs, real=True)
    fid_tmp.update(sub_gen_imgs, real=False)
    current_fid = float(fid_tmp.compute())
    
    diff = current_fid - baseline_fid
    
    # 표 안에 들어갈 이미지 태그 및 데이터 추가
    img_tag = f"<img src='{target_dir}/{filename}' width='150'>"
    md_content += f"| {img_tag} | `{filename}` | **{current_fid:.4f}** | **{diff:+.4f}** |\n"

# 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f"✅ 계산 완료! 시각화 리포트가 '{output_file}'로 저장되었습니다.")
