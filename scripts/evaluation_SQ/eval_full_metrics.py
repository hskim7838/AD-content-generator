import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# DINO 모델만 사용 (LPIPS는 패키지 문제로 제외)
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

target_dir = 'output_demo/output_image/extra_score_data'
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
ref_feat = get_dino_features(os.path.join(target_dir, filenames[0]))

print("🚀 DINO 통합 평가 시작...")
md_lines = ["# 📊 최종 품질 평가 리포트", "", "| 파일명 | DINO Similarity |", "|:---|:---:|"]
for f in filenames[1:]:
    feat = get_dino_features(os.path.join(target_dir, f))
    sim = F.cosine_similarity(ref_feat.unsqueeze(0), feat.unsqueeze(0)).item()
    md_lines.append(f"| `{f}` | {sim:.4f} |")

with open('full_report.md', 'w') as f:
    f.write("\n".join(md_lines))
print("✅ 'full_report.md'가 생성되었습니다.")
