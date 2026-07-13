# -*- coding: utf-8 -*-
"""
step5_random_batch_pipeline.py
---------------------------------
[5단계] 랜덤 조합 9개 -> compare_models() 실제 GPT 호출 -> 수치 옵션을 반영한 배너 합성까지 한 번에.

경로는 config.py에서 가져옵니다 — 이미지 경로를 바꿨다면 config.py만 수정하면
step1~5 전체(이 파일 포함)에 반영됩니다.

사전 조건:
  1) .env 파일에 OPENAI_API_KEY가 설정되어 있어야 함
  2) step1, step4를 먼저 실행해 tiny.json / option_pools.json이 준비되어 있어야 함
"""

import os
import random

import pandas as pd
from dotenv import load_dotenv

import config
from banner_utils import compose_banner
from compare_copy_llms import BackendConfig, compare_models
from data_prep import generate_combos, load_option_pools

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

random.seed(42)

# 조합마다 API를 여러 모델 다 부르면 비용/시간이 배로 늘어나므로 기본은 1개 모델만 사용.
# 여러 모델을 비교하고 싶으면 BACKENDS_FOR_LOOP에 항목을 추가하세요.
BACKENDS_FOR_LOOP = [
    BackendConfig(name="GPT-5.4 mini", provider="openai_compatible", model="gpt-5.4-mini"),
]


def main(n_combos: int = 9):
    if not os.path.exists(config.PRODUCT_ATTRS_PATH):
        raise FileNotFoundError(
            f"{config.PRODUCT_ATTRS_PATH} 가 없습니다. step1_prepare_tiny_json.py를 먼저 실행하세요."
        )
    if not os.path.exists(config.OPTION_POOLS_PATH):
        raise FileNotFoundError(
            f"{config.OPTION_POOLS_PATH} 가 없습니다. step4_config_based_run.py를 먼저 실행하세요."
        )

    os.makedirs(config.RANDOM_BATCH_OUTPUT_DIR, exist_ok=True)

    import json
    with open(config.PRODUCT_ATTRS_PATH, encoding="utf-8") as f:
        product_attrs = json.load(f)

    combos = generate_combos(n_combos, json_path=config.OPTION_POOLS_PATH)
    for i, c in enumerate(combos, 1):
        print(f"[{i}] {c}")

    store_info = load_option_pools(config.OPTION_POOLS_PATH)["store_info"]

    print(f"\n=== {n_combos}개 조합 실제 GPT 호출 + 배너 합성 시작 ===")
    results = []
    for i, combo in enumerate(combos, 1):
        print(f"\n[{i}/{n_combos}] tone={combo['tone']} product={combo['product']} ...")

        df = compare_models(
            product_attrs,
            store_info,
            tone=combo["tone"],
            product=combo["product"],
            brand=combo["brand"],
            sales=combo["sales"],
            focusing_degree=combo["focusing_degree"],
            background=None,
            background_intensity=combo["background_intensity"],
            preservation_degree=combo["preservation_degree"],
            backends=BACKENDS_FOR_LOOP,
        )

        row = df.iloc[0]
        if row.get("error"):
            print(f"  ❌ GPT 호출 실패: {row['error']} -> 이 조합은 건너뜁니다.")
            continue

        copy = {
            "model": row["model"],
            "title": row["title"],
            "subtitle": row["subtitle"],
            "price": row["price"],
            "cta": row["cta"],
        }

        out_path = os.path.join(config.RANDOM_BATCH_OUTPUT_DIR, f"banner_{i:02d}.jpg")
        compose_banner(config.PRODUCT_IMAGE_PATH, copy, combo, out_path)
        print(f"  ✅ 배너 생성 완료 -> {out_path}  (title: {copy['title']})")

        results.append({**combo, **copy, "image_path": out_path})

    print("\n=== 전체 완료 ===")
    print(f"성공: {len(results)} / {n_combos}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    results_df = main()
    print(results_df)
