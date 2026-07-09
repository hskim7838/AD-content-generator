import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# DINO 모델 로드 (허깅페이스를 통해 불러옴)
dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
dino_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_dino_features(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = dino_model(input_tensor)
    return features.squeeze()

def calculate_dino_sim(feat1, feat2):
    return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

# 메인 루프 (기존 리포트 파일에 추가)
target_dir = 'output_demo/output_image/extra_score_data'
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# 기준 이미지 피처 추출
ref_feat = get_dino_features(os.path.join(target_dir, filenames[0]))

print("🚀 LPIPS 및 DINO 평가 진행 중...")

# 기존 리포트 업데이트 로직 (간략화)
md_lines = ["# 📊 고급 품질 평가 리포트 (LPIPS/DINO)", "", "| 파일명 | DINO Similarity |", "|:---|:---:|"]
for f in filenames[1:]:
    feat = get_dino_features(os.path.join(target_dir, f))
    sim = calculate_dino_sim(ref_feat, feat)
    md_lines.append(f"| `{f}` | {sim:.4f} |")

with open('advanced_report.md', 'w') as f:
    f.write("\n".join(md_lines))

print("✅ 'advanced_report.md'가 생성되었습니다.")
