# 광고 배너 자동 생성 파이프라인 (VSCode / JupyterLab용)

Colab 노트북(`copywriting_4_수정(2).py`)에 있던 Google Drive 마운트, `userdata`,
`!pip install`, `%%writefile` 등 Colab 전용 요소를 모두 제거하고, 로컬 환경에서
바로 돌아가는 모듈/스크립트로 분리했습니다.

## 파일 구성

```
config.py                      # ★ 경로/상품정보를 관리하는 단일 설정 파일 — 여기만 고치면 됨
compare_copy_llms.py           # GPT 계열 모델로 카피(제목/서브카피/가격/CTA) 생성
data_prep.py                   # tiny.json 저장, option_pools/run_config 자동 생성, 랜덤 조합
banner_utils.py                # 폰트 탐색, 그라데이션, 텍스트 줄바꿈, 배너 합성(심플/수치옵션 버전)

step1_prepare_tiny_json.py     # [1단계] 이미지+캡션 -> tiny_dataset/ 저장
step2_generate_copy.py         # [2단계] compare_models() -> llm_output.json
step3_compose_banners_simple.py# [3단계] llm_output.json -> 배너 이미지(심플)
step4_config_based_run.py      # [4단계] run_config.json/option_pools.json 기반 실행
step5_random_batch_pipeline.py # [5단계] 랜덤 조합 9개 -> GPT 호출 -> 수치옵션 반영 배너

notebook_main.py                # 위 1~5단계를 "# %%" 셀로 이어붙인 통합 스크립트
requirements.txt
.env.example
```

## 경로는 이제 config.py 한 곳에서만 관리합니다

이전 버전은 각 step 파일에 이미지 경로가 따로 하드코딩돼 있어서, step1만 고치고
step5는 안 고치면 기본값(`./bakery.png`)으로 계속 실행되는 문제가 있었습니다.
지금은 `config.py`의 `PRODUCT_IMAGE_PATH` **한 줄만** 수정하면 step1~5, notebook_main.py
전부에 동일하게 반영됩니다.

```python
# config.py
PRODUCT_IMAGE_PATH = "./bakery.png"   # <- 본인 이미지 경로로 이 줄만 수정
```

## 실행 순서

1. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

2. API 키 설정
   ```bash
   cp .env.example .env
   # .env를 열어 OPENAI_API_KEY=sk-... 로 교체
   ```

3. `config.py`에서 `PRODUCT_IMAGE_PATH`를 실제 이미지 경로로 수정 (한 곳만 고치면 끝)

4. 순서대로 실행
   ```bash
   python step1_prepare_tiny_json.py       # tiny_dataset/ 생성
   python step2_generate_copy.py           # llm_output.json 생성 (GPT 호출)
   python step3_compose_banners_simple.py  # ad_banners/*.jpg 생성
   python step4_config_based_run.py        # run_config/option_pools 자동 생성 + 실행
   python step5_random_batch_pipeline.py   # 랜덤 조합 9개 배치 (비용/시간 큼, 선택)
   ```
   step2/step4/step5는 실행 전 필요한 파일(tiny.json, option_pools.json)이
   없으면 "어느 step을 먼저 실행해야 하는지" 에러 메시지로 바로 알려줍니다.

## VSCode에서 셀 단위로 실행하기

`notebook_main.py`를 열면 Python 확장이 `# %%` 를 셀 경계로 인식해서
각 블록 위에 **Run Cell**이 뜹니다. IPython 매직은 전혀 없어서
`python notebook_main.py`로 통째 실행도 됩니다.

## JupyterLab에서 노트북(.ipynb)으로 열기

```bash
pip install jupytext            # requirements.txt에 이미 포함
jupytext --to notebook notebook_main.py
```

생성된 `notebook_main.ipynb`를 JupyterLab에서 열면 `# %%` 기준으로 셀이 나뉘어 표시됩니다.

## Colab 버전과 달라진 점

| Colab | 로컬(VSCode/JupyterLab) 버전 |
|---|---|
| `drive.mount('/content/drive')` | 제거 — 로컬 파일 경로를 직접 사용 |
| `google.colab.userdata.get(...)` | `.env` 파일 + `python-dotenv` |
| `!pip install ...` | `requirements.txt` |
| 경로가 여러 파일에 흩어져 하드코딩 | `config.py` 한 곳으로 통합 |
| `%%writefile compare_copy_llms.py` (주석 처리되어 실행 안 됨) | 실제 `compare_copy_llms.py` 파일로 복원 |
