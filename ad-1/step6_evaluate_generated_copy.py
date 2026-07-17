# -*- coding: utf-8 -*-
"""
step6_evaluate_generated_copy.py
------------------------------------
step1_generate_copy.py가 생성한 ./generated_copy_results.json 을 읽어서
eval_copy.evaluate_dataframe()로 정량 평가를 붙이는 스크립트.

STEP1과 파이썬 프로세스 자체가 분리되어 있으므로, 변수 공유 없이
generated_copy_results.json 파일을 통해서만 데이터를 주고받는다.
이 파일에는 상품별 product_attrs/store_info("products")와 각 행이 어느 상품
소속인지("product_id")가 함께 들어있어, 상품이 여러 개(=tiny.json이 여러 개)여도
행마다 올바른 원본 정보와 대조해서 평가할 수 있다.

실행 순서:
  1) python copywrite_step4_5.py               (generated_copy_results.json 생성)
  2) python step6_evaluate_generated_copy.py    (이 파일 — 정량 평가 + CSV 저장)
  3) python step7_visualize_eval_result.py      (막대그래프 PNG 저장)

사전 조건: eval_copy.py 가 이 파일과 같은 폴더에 있어야 함.

포함되는 평가 컬럼:
  latency_sec                                  (STEP1의 compare_models() 응답 시간)
  rule_pass, rule_violations                   (규칙 기반, API 호출 없음)
  faithfulness_score                            (임베딩 유사도)
  judge_tone_fit, judge_persuasiveness,
  judge_naturalness, judge_constraint_adherence,
  judge_total                                   (LLM judge 채점)
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv

from eval_copy import evaluate_dataframe, show_full

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

GENERATED_COPY_RESULTS_PATH = "./generated_copy_results.json"
EVAL_RESULT_CSV_PATH = "./generated_copy_eval_result.csv"

if not os.path.exists(GENERATED_COPY_RESULTS_PATH):
    raise FileNotFoundError(
        f"{GENERATED_COPY_RESULTS_PATH} 가 없습니다. copywrite_step4_5.py를 먼저 실행하세요."
    )

# ── 1. STEP1이 저장해둔 카피 결과 + 상품별 원본 정보 불러오기 ──────────
with open(GENERATED_COPY_RESULTS_PATH, encoding="utf-8") as f:
    payload = json.load(f)

products = payload["products"]   # product_id -> {"product_attrs":..., "store_info":...}
results = payload["results"]     # 각 행에 product_id 포함

print(f"불러온 상품 수: {len(products)}개 / 카피 결과: {len(results)}개")
if not results:
    raise RuntimeError("generated_copy_results.json 안에 평가할 카피가 없습니다.")

# ── 2. 정량 평가 실행 (상품 × 톤 조합별로 나눠서 evaluate_dataframe 호출) ──
# faithfulness_score/rule_based_check가 그 상품(tiny.json)의 product_attrs/store_info와
# 대조되도록, 상품과 톤이 같은 행끼리 묶어서 평가한다.
raw_df = pd.DataFrame(results)

evaluated_parts = []
for (product_id, tone), part in raw_df.groupby(["product_id", "tone"]):
    product_info = products.get(product_id)
    if product_info is None:
        print(f"⚠ products에 없는 product_id={product_id} 행 건너뜀")
        continue
    evaluated = evaluate_dataframe(
        part,
        product_info["product_attrs"],
        product_info["store_info"],
        tone=tone,
        debug=True,
    )
    evaluated_parts.append(evaluated)
eval_result_df = pd.concat(evaluated_parts, ignore_index=True)

# ── 3. 결과 출력 ────────────────────────────────────────────────────
DISPLAY_COLUMNS = [
    "product_id", "model", "tone", "title", "subtitle", "cta",
    "latency_sec",
    "rule_pass", "rule_violations",
    "faithfulness_score",
    "judge_tone_fit", "judge_persuasiveness", "judge_naturalness",
    "judge_constraint_adherence", "judge_total",
]

print("\n=== [STEP 6] 조합별 정량 평가 결과 ===")
print(show_full(eval_result_df)[DISPLAY_COLUMNS].sort_values("judge_total", ascending=False))

print("\n=== [STEP 6] 모델별 평균 (지연시간 포함) ===")
print(eval_result_df.groupby("model")[[
    "latency_sec", "faithfulness_score",
    "judge_tone_fit", "judge_persuasiveness", "judge_naturalness",
    "judge_constraint_adherence", "judge_total",
]].mean().round(3))

print("\n=== [STEP 6] 톤별 평균 ===")
print(eval_result_df.groupby("tone")[[
    "faithfulness_score",
    "judge_tone_fit", "judge_persuasiveness", "judge_naturalness",
    "judge_constraint_adherence", "judge_total",
]].mean().round(3))

print("\n=== [STEP 6] 상품별 평균 ===")
print(eval_result_df.groupby("product_id")[[
    "faithfulness_score",
    "judge_tone_fit", "judge_persuasiveness", "judge_naturalness",
    "judge_constraint_adherence", "judge_total",
]].mean().round(3))

# ── 4. CSV 저장 ──────────────────────────────────────────────────────
eval_result_df.to_csv(EVAL_RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\n저장 완료 -> {EVAL_RESULT_CSV_PATH}")
print("시각화는 `python step7_visualize_eval_result.py` 를 실행하세요.")
