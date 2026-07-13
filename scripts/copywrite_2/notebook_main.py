# -*- coding: utf-8 -*-
"""
notebook_main.py
------------------
1~5단계를 "# %%" 셀 마커로 이어붙인 통합 스크립트.
경로는 config.py 한 곳에서만 관리합니다 (PRODUCT_IMAGE_PATH를 바꾸고 싶으면 config.py만 수정).

- VSCode: 파일을 열면 Python 확장이 "# %%" 를 셀로 인식해서 각 블록 위에 "Run Cell"이 뜹니다.
- JupyterLab: `jupytext --to notebook notebook_main.py` 로 변환 후 .ipynb로 열면 됩니다.
- 터미널: `python notebook_main.py` 로 통째 실행도 가능합니다.
"""

# %%
# ── 0. 환경 설정 ───────────────────────────────────────────────
import os

from dotenv import load_dotenv

import config

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 값을 채워주세요.")

# %%
# ── 1. 원본 데이터 준비 (tiny.json 생성) ───────────────────────────
from data_prep import save_tiny_dataset

tiny_data_list = [
    {
        "id": config.PRODUCT_ID,
        "image": config.PRODUCT_IMAGE_PATH,
        "new_caption": config.PRODUCT_CAPTION,
        "store_info": config.STORE_INFO_RAW,
    },
]
save_tiny_dataset(tiny_data_list, img_dir=config.IMG_DIR, json_dir=config.JSON_DIR)

# %%
# ── 2. 카피 생성 (compare_models) ─────────────────────────────────
import json

from compare_copy_llms import compare_models

with open(config.PRODUCT_ATTRS_PATH, encoding="utf-8") as f:
    product_attrs = json.load(f)

df = compare_models(product_attrs, config.STORE_INFO, tone="친근한", product="크루아상 & 콜드브루 세트")
print(df[["model", "tone", "title", "subtitle", "price", "cta"]])

copy_results = df[["model", "tone", "title", "subtitle", "price", "cta"]].to_dict(orient="records")
with open(config.LLM_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(copy_results, f, ensure_ascii=False, indent=4)

# %%
# ── 3. 배너 합성 (심플 버전) ────────────────────────────────────────
from banner_utils import compose_banner_simple

os.makedirs(config.BANNER_OUTPUT_DIR, exist_ok=True)

for copy in copy_results:
    out_name = f"banner_{copy['model'].replace(' ', '_').replace('.', '')}.jpg"
    out_path = os.path.join(config.BANNER_OUTPUT_DIR, out_name)
    compose_banner_simple(config.PRODUCT_IMAGE_PATH, copy, out_path)
    print(f"  ✅ [{copy['model']}] 배너 생성 완료 -> {out_path}")

# %%
# ── 4. option_pools / run_config 자동 생성 + 설정 기반 실행 ──────────
from data_prep import build_configs_from_tiny_json
from step4_config_based_run import run_compare_models_from_config

build_configs_from_tiny_json(
    config.PRODUCT_ATTRS_PATH,
    option_pools_path=config.OPTION_POOLS_PATH,
    run_config_path=config.RUN_CONFIG_PATH,
    overwrite=False,
)
cfg_df = run_compare_models_from_config()
print(cfg_df)

# %%
# ── 5. 랜덤 조합 9개 실전 배치 (비용/시간 소요 큼 — 필요할 때만 실행) ──
from step5_random_batch_pipeline import main as run_random_batch

results_df = run_random_batch(n_combos=9)
print(results_df)
