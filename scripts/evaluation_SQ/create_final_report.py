import pandas as pd
import os

# 각 파일 경로
files = {
    'basic': 'evaluation_report.csv',  # PSNR, SSIM, LPIPS 포함
    'fid': 'fid_report.md',           # FID가 들어있는 파일 (내용 파싱 필요)
    'clip': 'clip_score_v2_report.csv' # CLIP Score 파일
}

# 1. 기본 지표 로드 (PSNR, SSIM, LPIPS)
df_main = pd.read_csv(files['basic'])

# 2. CLIP Score 로드 및 병합
df_clip = pd.read_csv(files['clip'])
df_final = pd.merge(df_main, df_clip, on='File')

# 3. FID는 전체 평균값이므로 별도 컬럼으로 추가 (필요시)
# 만약 FID값이 텍스트 파일에 있다면 값을 직접 입력하거나 파싱합니다.
df_final['FID'] = 15.2  # 예시 값입니다. 실제 fid_report.md의 값을 넣어주세요.

# 저장
df_final.to_csv('final_evaluation_summary.csv', index=False)
print("통합 리포트 'final_evaluation_summary.csv'가 성공적으로 저장되었습니다!")