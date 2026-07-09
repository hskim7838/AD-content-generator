import pandas as pd
import os

# 1. 데이터 로드 및 병합
df_main = pd.read_csv('final_evaluation_summary.csv')
try:
    df_clip = pd.read_csv('clip_score_v2_report.csv')
    if 'image_name' in df_clip.columns:
        df_main = pd.merge(df_main, df_clip[['image_name', 'clip_score']], on='image_name', how='left')
except Exception:
    pass

# 2. HTML 구조 생성
html_str = """<html><head><meta charset="utf-8"></head><body>
<table border='1' style='border-collapse: collapse; width: 100%; text-align: center; font-family: sans-serif;'>
    <tr style='background-color: #f2f2f2; font-weight: bold;'>
        <th style='padding: 10px;'>미리보기</th>
        <th>파일명</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>FID 영향도</th>
        <th>CLIP Score</th>
        <th>DINO Sim</th>
        <th>LPIPS</th>
    </tr>
"""
for idx, row in df_main.iterrows():
    img_name = row.get('image_name', '')
    psnr = f"{row.get('psnr', '-'):.2f}" if pd.notnull(row.get('psnr')) else "-"
    ssim = f"{row.get('ssim', '-'):.4f}" if pd.notnull(row.get('ssim')) else "-"
    fid = f"{row.get('fid_impact', '-'):+}" if pd.notnull(row.get('fid_impact')) else "-"
    clip = f"{row.get('clip_score', '-'):.4f}" if pd.notnull(row.get('clip_score')) else "-"
    dino = f"{row.get('dino_similarity', '-'):.4f}" if pd.notnull(row.get('dino_similarity')) else "-"
    lpips = f"{row.get('lpips', '-'):.4f}" if pd.notnull(row.get('lpips')) else "-"
    
    html_str += f"""
    <tr>
        <td style='padding: 5px;'><img src='./{img_name}' width='100'></td>
        <td style='text-align: left; padding-left: 10px;'><code>{img_name}</code></td>
        <td>{psnr}</td>
        <td>{ssim}</td>
        <td>{fid}</td>
        <td style='background-color: #f5f5f5; font-weight: bold;'>{clip}</td>
        <td>{dino}</td>
        <td>{lpips}</td>
    </tr>"""
html_str += "</table></body></html>"

with open("comprehensive_report.html", "w", encoding="utf-8") as f:
    f.write(html_str)
print("★ 종합 리포트 파일(comprehensive_report.html) 생성 완료! ★")
