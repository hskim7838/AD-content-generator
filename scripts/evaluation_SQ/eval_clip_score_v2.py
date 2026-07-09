import os
import torch
import clip
from PIL import Image
import pandas as pd

# 설정
DATA_DIR = '/home/ai3/AD-content/test_data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CLIP 모델 로드
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_score(img_path, text):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_input)
        
        # 코사인 유사도 계산
        score = (image_features @ text_features.T).item()
    return score

# 평가 수행
results = []
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])

# 주의: 여기서는 파일명에 프롬프트 정보가 포함되어 있다고 가정합니다.
# 실제 프롬프트 파일이 따로 있다면 로직을 수정해야 합니다.
for file in files:
    prompt = "A high quality image" # 실제 프롬프트로 교체 필요
    score = get_clip_score(os.path.join(DATA_DIR, file), prompt)
    results.append({'File': file, 'CLIP_Score': score})

df = pd.DataFrame(results)
df.to_csv('clip_score_v2_report.csv', index=False)
print("CLIP Score 평가 완료! clip_score_v2_report.csv가 생성되었습니다.")