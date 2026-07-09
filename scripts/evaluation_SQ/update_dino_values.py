import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 모델 로드
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

# 리포트 파일 읽기 및 값 교체
with open('full_report.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
file_idx = 0
for line in lines:
    if "| <img" in line and file_idx < len(filenames) - 1:
        file_idx += 1
        curr_path = os.path.join(target_dir, filenames[file_idx])
        feat = get_dino_features(curr_path)
        sim = F.cosine_similarity(ref_feat.unsqueeze(0), feat.unsqueeze(0)).item()
        # '계산중'을 계산된 값으로 교체
        line = line.replace("계산중", f"{sim:.4f}")
    new_lines.append(line)

with open('full_report.md', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("✅ full_report.md의 DINO Sim 점수가 업데이트되었습니다.")
