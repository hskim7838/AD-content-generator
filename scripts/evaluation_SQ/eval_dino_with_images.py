import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# DINO 모델 설정
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dino_features(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        return dino_model(input_tensor).squeeze()

# 데이터 경로 설정
target_dir = 'output_demo/output_image/extra_score_data'
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not filenames:
    print("❌ 이미지를 찾을 수 없습니다.")
    exit()

# 기준 이미지 피처 추출
ref_path = os.path.join(target_dir, filenames[0])
ref_feat = get_dino_features(ref_path)

print("🚀 이미지와 함께 DINO Similarity 평가 진행 중...")

# 마크다운 리포트 작성 (이미지 태그 포함)
md_content = "# 📊 최종 품질 평가 리포트 (이미지 포함)\n\n"
md_content += f"- **기준 원본 이미지:** `{filenames[0]}` <br>\n"
md_content += f"<img src='./{ref_path}' width='200'>\n\n"
md_content += "| 미리보기 | 파일명 | DINO Similarity |\n"
md_content += "|:---:|---|:---:|\n"

# 생성 이미지들에 대해 루프 실행
for f in filenames[1:]:
    curr_path = os.path.join(target_dir, f)
    feat = get_dino_features(curr_path)
    sim = F.cosine_similarity(ref_feat.unsqueeze(0), feat.unsqueeze(0)).item()
    
    # 이미지 태그 생성 (주피터랩 상대 경로 매핑)
    img_tag = f"<img src='./{curr_path}' width='100'>"
    
    md_content += f"| {img_tag} | `{f}` | **{sim:.4f}** |\n"

# 파일로 저장
with open('full_report.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print("✅ 이미지 시각화가 포함된 'full_report.md'가 생성되었습니다.")
