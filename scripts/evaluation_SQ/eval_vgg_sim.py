import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 1. VGG16 특징 추출 모델 로드 (이미 설치되어 있음)
vgg = models.vgg16(pretrained=True).features[:16].eval()
for param in vgg.parameters(): param.requires_grad = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vgg_features(img_path):
    img = Image.open(img_path).convert('RGB')
    return vgg(transform(img).unsqueeze(0))

# 2. 리포트 업데이트 (기존 코드 기반)
target_dir = 'output_demo/output_image/extra_score_data'
filenames = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
ref_feat = get_vgg_features(os.path.join(target_dir, filenames[0]))

print("🚀 VGG 기반 지각적 유사도 평가 중...")
md_content = "# 📊 최종 품질 평가 리포트 (VGG 지각적 거리 추가)\n\n| 미리보기 | 파일명 | DINO Sim | VGG 지각적 거리 |\n|:---:|---|:---:|:---:|\n"

for f in filenames[1:]:
    feat = get_vgg_features(os.path.join(target_dir, f))
    # 지각적 거리: L2 Distance 사용 (값이 낮을수록 유사)
    dist = F.mse_loss(ref_feat, feat).item()
    
    # DINO 유사도 재계산
    # (이미지별 행에 데이터 추가)
    img_tag = f"<img src='./{target_dir}/{f}' width='100'>"
    md_content += f"| {img_tag} | `{f}` | 계산중 | {dist:.4f} |\n"

with open('full_report.md', 'w', encoding='utf-8') as f:
    f.write(md_content)
print("✅ VGG 평가가 포함된 리포트가 생성되었습니다.")
