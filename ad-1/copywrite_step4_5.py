# -*- coding: utf-8 -*-
"""copywrite_step4_5.py (VSCode/로컬 실행용)

이미지 스캔·캡션 자동생성·option_pools 생성(구 STEP1~3)은 이미 완료되어
아래 파일들이 존재한다고 가정하고, 그 다음 단계만 수행하는 버전입니다.

  - ./option_pools.json   (tiny.json들을 모아서 만든 매장/옵션 풀)
  - ./run_config.json     (run_config_source_id 기준 대표 상품 설정)
  - run_config.json 안 product_attrs_path가 가리키는 tiny.json(id.json) 원본

즉 tiny_dataset/images, tiny_dataset/jsons, option_pools.json, run_config.json이
전부 이미 만들어져 있는 상태에서, 카피 문구 생성만 다시 돌리고 싶을 때 쓰는 파일입니다.
(이미지/캡션을 새로 만들거나 option_pools.json을 다시 만들고 싶으면 예전 STEP1~3
코드가 필요합니다.)

전체 파이프라인 순서:
  1) (완료됨) 이미지 -> tiny.json -> option_pools.json / run_config.json
  2) python copywrite_step4_5.py          (이 파일 — 카피 문구 9개 생성)
  3) python step6_evaluate_generated_copy.py
  4) python step7_visualize_eval_result.py
  5) python step8_compose_banner.py       (평가 결과 기반으로 이미지에 카피 합성)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ── 경로 기준점 ──────────────────────────────────────────────────────
# VSCode 실행 버튼/디버거는 터미널 cwd가 이 파일 위치와 다를 수 있어서,
# "현재 작업 디렉터리(cwd)"가 아니라 "이 스크립트 파일이 있는 폴더"를
# 모든 상대경로의 기준으로 고정한다.
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    """절대경로면 그대로, 상대경로면 BASE_DIR 기준으로 변환."""
    p = Path(path_str)
    return str(p if p.is_absolute() else (BASE_DIR / p))


load_dotenv(BASE_DIR / ".env")  # 프로젝트 루트(스크립트 위치)의 .env에서 OPENAI_API_KEY를 읽어옴

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY=sk-... 를 추가하거나 "
        "os.environ['OPENAI_API_KEY']를 직접 설정한 뒤 다시 실행하세요."
    )

OPTION_POOLS_PATH = str(BASE_DIR / "option_pools.json")
RUN_CONFIG_PATH = str(BASE_DIR / "run_config.json")

for _p in (OPTION_POOLS_PATH, RUN_CONFIG_PATH):
    if not os.path.exists(_p):
        raise FileNotFoundError(
            f"{_p} 가 없습니다. 이미지 스캔/캡션 생성/옵션 풀 생성(구 STEP1~3)을 "
            "먼저 완료해서 tiny_dataset/, option_pools.json, run_config.json을 만들어두세요."
        )


# ══════════════════════════════════════════════════════════════
# STEP 1 (구 STEP4). compare_copy_llms.py 모듈 파일 생성
# --------------------------------------------------------------
# "광고 문구(카피라이팅) 생성" 용도로 GPT 계열 모델에 같은 입력(product_attrs + store_info)을
# 실제로 돌려서 카피(JSON)를 생성하는 모듈.
#
# [옵션]
# - product / brand / sales       : 카피 생성 프롬프트에 실제 반영되는 상품/브랜드/판매 정보
# - focusing_degree (0.0~1.0)     : 제품 강조(포커스) 정도. 높을수록 제품이 선명/부각됨
# - background (텍스트)            : 원하는 배경 컨셉 설명
# - background_intensity (0.0~1.0): 배경의 존재감/복잡도. 낮으면 심플·단색에 가까움
# - preservation_degree (0.0~1.0) : 원본 이미지 보존 정도. 높을수록 원본 형태를 최대한 유지
#
# focusing_degree / background_intensity / preservation_degree 는 카피 생성 자체보다는
# "이후 ControlNet/SD 이미지 생성 단계"에서 쓰일 파라미터입니다.
# to_image_gen_params()로 실제 SD/ControlNet 파라미터로 변환할 수 있습니다.
# ══════════════════════════════════════════════════════════════
_COMPARE_COPY_LLMS_SOURCE = r'''
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

ToneType = Literal["미니멀", "강조형", "친근한", "고급스러운", "위트있는"]


def _clamp01(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}는 0.0~1.0 사이의 값이어야 합니다. (입력값: {value})")
    return float(value)


class AdCopy(BaseModel):
    tone: ToneType = Field(description="이 카피가 지향하는 톤앤매너")
    title: str = Field(description="배너 메인 타이틀. 12자 내외")
    subtitle: str = Field(description="서브 카피. 20자 내외")
    price_text: Optional[str] = Field(default=None, description="가격 강조 문구. 없으면 null")
    cta: str = Field(description="행동 유도 문구(CTA)")


SYSTEM_PROMPT = """\
당신은 오프라인 소상공인을 위한 광고 카피라이터입니다.
- 타이틀은 25자 내외, 서브카피는 35자 내외로 간결하게 작성합니다.
- 주어진 상품/매장 정보에 없는 사실을 지어내지 않습니다.
- 반드시 아래 JSON 스키마 형식으로만 답하세요. 다른 설명은 절대 붙이지 마세요.

{schema}
"""


def _build_user_prompt(
    product_attrs: dict,
    store_info: dict,
    tone: str,
    product: Optional[str] = None,
    brand: Optional[str] = None,
    sales: Optional[str] = None,
    background: Optional[str] = None,
) -> str:
    merged_store_info = dict(store_info)
    if product:
        merged_store_info.setdefault("product", product)
    if brand:
        merged_store_info.setdefault("brand", brand)
    if sales:
        merged_store_info.setdefault("sales", sales)

    background_note = f"\n[배경 컨셉 참고]\n{background}\n" if background else ""

    return f"""\
[상품 속성 (tiny.json)]
{json.dumps(product_attrs, ensure_ascii=False, indent=2)}

[매장 정보]
{json.dumps(merged_store_info, ensure_ascii=False, indent=2)}
{background_note}
[요청]
톤앤매너 "{tone}" 에 맞는 광고 카피 1세트를 생성해 주세요.
"""


@dataclass
class BackendConfig:
    name: str
    provider: Literal["openai_compatible"]
    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"


BACKENDS: List[BackendConfig] = [
    BackendConfig(name="GPT-5.5", provider="openai_compatible", model="gpt-5.5"),
    BackendConfig(name="GPT-5.4 mini", provider="openai_compatible", model="gpt-5.4-mini"),
    BackendConfig(name="GPT-5.4 nano", provider="openai_compatible", model="gpt-5.4-nano"),
]


def _call_openai_compatible(cfg: BackendConfig, system: str, user: str) -> str:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} 환경변수가 설정되지 않았습니다.")
    client = OpenAI(api_key=api_key, base_url=cfg.base_url)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def _call_backend(cfg: BackendConfig, system: str, user: str) -> str:
    if cfg.provider == "openai_compatible":
        return _call_openai_compatible(cfg, system, user)
    raise ValueError(f"알 수 없는 provider: {cfg.provider}")


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def to_image_gen_params(
    focusing_degree: float,
    background_intensity: float,
    preservation_degree: float,
) -> dict:
    """
    0.0~1.0 수치 옵션을 실제 Stable Diffusion / ControlNet 파라미터로 변환.

    - controlnet_conditioning_scale : 원본 제품 형태를 얼마나 강하게 따를지 (0~2 범위가 일반적)
        preservation_degree가 높을수록 원본 형태를 더 강하게 유지 -> 값이 커짐
    - denoising_strength            : img2img에서 원본을 얼마나 새로 그릴지 (0=원본 그대로, 1=완전 재생성)
        preservation_degree와 반대 방향 -> 1 - preservation_degree
    - guidance_scale                : 프롬프트를 얼마나 강하게 따를지. focusing_degree가 높을수록
        제품/디테일을 선명하게 살리기 위해 값을 올림 (통상 5~12 범위로 매핑)
    - background_complexity_hint    : 배경 프롬프트에 얼마나 디테일/오브젝트를 추가할지 (0=단색/심플, 1=복잡한 배경)
    - depth_of_field                : focusing_degree가 높을수록 배경을 더 흐리게(아웃포커스) 처리
    """
    focusing_degree = _clamp01(focusing_degree, "focusing_degree")
    background_intensity = _clamp01(background_intensity, "background_intensity")
    preservation_degree = _clamp01(preservation_degree, "preservation_degree")

    return {
        "controlnet_conditioning_scale": round(0.4 + preservation_degree * 1.2, 2),  # 0.4 ~ 1.6
        "denoising_strength": round(1.0 - preservation_degree, 2),                    # 1.0 ~ 0.0
        "guidance_scale": round(5.0 + focusing_degree * 7.0, 2),                      # 5.0 ~ 12.0
        "background_complexity_hint": round(background_intensity, 2),                # 0.0 ~ 1.0
        "depth_of_field": round(focusing_degree, 2),                                  # 0.0(전체 선명) ~ 1.0(배경 아웃포커스 강함)
    }


def compare_models(
    product_attrs: dict,
    store_info: dict,
    tone: str = "친근한",
    # ── 카피 생성에 직접 반영되는 옵션 ──────────────────────────
    product: Optional[str] = None,
    brand: Optional[str] = None,
    sales: Optional[str] = None,
    # ── 이후 이미지 생성(ControlNet 등) 단계로 전달되는 수치 옵션 (모두 0.0~1.0) ──
    focusing_degree: float = 0.5,
    background: Optional[str] = None,
    background_intensity: float = 0.5,
    preservation_degree: float = 0.5,
    backends: Optional[List[BackendConfig]] = None,
    max_retries: int = 1,
) -> pd.DataFrame:
    focusing_degree = _clamp01(focusing_degree, "focusing_degree")
    background_intensity = _clamp01(background_intensity, "background_intensity")
    preservation_degree = _clamp01(preservation_degree, "preservation_degree")

    image_gen_params = to_image_gen_params(
        focusing_degree, background_intensity, preservation_degree
    )

    backends = backends or BACKENDS
    schema_hint = json.dumps(AdCopy.model_json_schema(), ensure_ascii=False, indent=2)
    system = SYSTEM_PROMPT.format(schema=schema_hint)
    user = _build_user_prompt(
        product_attrs, store_info, tone,
        product=product, brand=brand, sales=sales, background=background,
    )

    rows = []
    for cfg in backends:
        row = {
            "model": cfg.name,
            "tone": tone,
            "title": None,
            "subtitle": None,
            "price": None,
            "cta": None,
            "latency_sec": None,
            "error": None,
            # 카피 생성 반영 옵션
            "product": product,
            "brand": brand,
            "sales": sales,
            # 이미지 생성 단계로 넘어갈 수치 옵션 원본값
            "focusing_degree": focusing_degree,
            "background": background,
            "background_intensity": background_intensity,
            "preservation_degree": preservation_degree,
            # 수치 옵션을 실제 SD/ControlNet 파라미터로 변환한 값
            **{f"gen__{k}": v for k, v in image_gen_params.items()},
        }
        start = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                raw = _call_backend(cfg, system, user)
                parsed = AdCopy.model_validate(_extract_json(raw))
                row.update({
                    "title": parsed.title,
                    "subtitle": parsed.subtitle,
                    "price": parsed.price_text or "",
                    "cta": parsed.cta,
                    "latency_sec": round(time.time() - start, 2),
                })
                last_err = None
                break
            except (json.JSONDecodeError, ValidationError, RuntimeError, Exception) as e:
                last_err = e
                continue
        if last_err is not None:
            row["error"] = f"{type(last_err).__name__}: {last_err}"
            row["latency_sec"] = round(time.time() - start, 2)
        rows.append(row)

    return pd.DataFrame(rows)
'''

_COMPARE_COPY_LLMS_PATH = BASE_DIR / "compare_copy_llms.py"
with open(_COMPARE_COPY_LLMS_PATH, "w", encoding="utf-8") as f:
    f.write(_COMPARE_COPY_LLMS_SOURCE)

print("\n=== [STEP 1] compare_copy_llms.py 모듈 파일 생성 완료 ===")

# python 스크립트를 `python /어딘가/copywrite_step4_5.py` 처럼 다른 위치에서 실행해도
# 방금 만든 compare_copy_llms.py(스크립트와 같은 폴더)를 import할 수 있도록 sys.path에 추가
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compare_copy_llms import compare_models, BackendConfig


# ══════════════════════════════════════════════════════════════
# STEP 2 (구 STEP5). 랜덤 조합 9개 -> compare_models() 실제 GPT 호출 -> 카피 문구 생성
# --------------------------------------------------------------
# - option_pools.json에서 tone/product/brand/sales 등을 랜덤 조합해 9개 생성
# - 조합 하나당 모델 1개씩만 호출하되, 3개 모델(GPT-5.5 / mini / nano)이
#   번갈아가며(라운드로빈) 골고루 섞이게 함
# - 생성 결과는 generated_copy_results.json으로 저장 (STEP6 정량평가, STEP8 배너 합성에서 사용)
# ══════════════════════════════════════════════════════════════
random.seed(42)

with open(RUN_CONFIG_PATH, encoding="utf-8") as f:
    _run_cfg_for_random = json.load(f)
PRODUCT_ATTRS_PATH = resolve_path(_run_cfg_for_random["product_attrs_path"])

if not os.path.exists(PRODUCT_ATTRS_PATH):
    raise FileNotFoundError(
        f"run_config.json이 가리키는 tiny.json이 없습니다: {PRODUCT_ATTRS_PATH}\n"
        "이미지/캡션 생성(구 STEP1~2)이 완료된 상태인지 확인하세요."
    )

with open(PRODUCT_ATTRS_PATH, encoding="utf-8") as f:
    product_attrs = json.load(f)

# 이 tiny.json의 id를 product_id로 사용 (STEP6/STEP8에서 상품별로 원본과 대조하기 위함)
PRODUCT_ID = product_attrs.get("id") or os.path.splitext(os.path.basename(PRODUCT_ATTRS_PATH))[0]
_raw_image_path = product_attrs.get("image")
PRODUCT_IMAGE_PATH = resolve_path(_raw_image_path) if _raw_image_path else None

BACKENDS_POOL = [
    BackendConfig(name="GPT-5.5", provider="openai_compatible", model="gpt-5.5"),
    BackendConfig(name="GPT-5.4 mini", provider="openai_compatible", model="gpt-5.4-mini"),
    BackendConfig(name="GPT-5.4 nano", provider="openai_compatible", model="gpt-5.4-nano"),
]


def pick_backend(index: int) -> list[BackendConfig]:
    """조합 인덱스(1부터 시작)를 받아 BACKENDS_POOL을 라운드로빈으로 순환 배정."""
    return [BACKENDS_POOL[(index - 1) % len(BACKENDS_POOL)]]


def load_option_pools(json_path: str = OPTION_POOLS_PATH) -> dict:
    """
    store_info / tone / product / brand / sales 옵션 풀을 JSON 파일에서 읽어온다.
    호출할 때마다 파일을 새로 읽으므로, JSON 내용을 바꾸면 재실행 시 바로 반영된다.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        pools = json.load(f)

    required_keys = ["store_info", "tones", "products", "brands", "sales"]
    missing = [k for k in required_keys if k not in pools]
    if missing:
        raise ValueError(f"'{json_path}'에 다음 키가 없습니다: {missing}")

    return pools


def generate_combos(n: int = 9, json_path: str = OPTION_POOLS_PATH) -> list[dict]:
    """
    랜덤 조합 n개를 생성. 호출될 때마다 json_path를 새로 읽으므로
    실행할 때마다 최신 JSON 옵션 풀 기준으로 조합이 만들어진다.
    """
    pools = load_option_pools(json_path)
    combos = []
    for _ in range(n):
        combos.append({
            "tone": random.choice(pools["tones"]),
            "product": random.choice(pools["products"]),
            "brand": random.choice(pools["brands"]),
            "sales": random.choice(pools["sales"]),
            "focusing_degree": round(random.uniform(0.0, 1.0), 2),
            "background_intensity": round(random.uniform(0.0, 1.0), 2),
            "preservation_degree": round(random.uniform(0.0, 1.0), 2),
        })
    return combos


# store_info도 매 실행 시 JSON 기준 최신값으로 갱신
store_info = load_option_pools()["store_info"]

combos = generate_combos(9)
print("\n=== [STEP 2] 랜덤 조합 9개 ===")
for i, c in enumerate(combos, 1):
    print(f"[{i}] {c}")

print("\n=== [STEP 2] 9개 조합 실제 GPT 호출 시작 ===")
results = []
for i, combo in enumerate(combos, 1):
    backend_for_this_combo = pick_backend(i)

    df = compare_models(
        product_attrs,
        store_info,
        tone=combo["tone"],
        product=combo["product"],
        brand=combo["brand"],
        sales=combo["sales"],
        focusing_degree=combo["focusing_degree"],
        background=None,  # 필요하면 조합별 배경 설명 텍스트도 추가 가능
        background_intensity=combo["background_intensity"],
        preservation_degree=combo["preservation_degree"],
        backends=backend_for_this_combo,
    )

    row = df.iloc[0]
    if row.get("error"):
        print(f"[{i}/9] ❌ GPT 호출 실패: {row['error']} -> 이 조합은 건너뜁니다.")
        continue

    copy = {
        "model": row["model"],
        "title": row["title"],
        "subtitle": row["subtitle"],
        "price": row["price"],
        "cta": row["cta"],
        "latency_sec": row["latency_sec"],
    }

    print(f"\n[{i}/9] model={copy['model']}  tone={combo['tone']}  product={combo['product']}")
    print(f"  title    : {copy['title']}")
    print(f"  subtitle : {copy['subtitle']}")
    print(f"  price    : {copy['price']}")
    print(f"  cta      : {copy['cta']}")
    print(f"  latency  : {copy['latency_sec']}초")

    results.append({"product_id": PRODUCT_ID, **combo, **copy})

print("\n=== [STEP 2] 전체 완료 ===")
print(f"성공: {len(results)} / 9")

results_df = pd.DataFrame(results)
print(results_df)

# ── 정량 평가(STEP6) / 배너 합성(STEP8)으로 넘겨주기 위해 결과를 JSON으로 저장 ──
# 이후 단계들은 이 파일과 별개의 프로세스로 실행되므로, 파이썬 변수를 공유하지 않고
# 파일을 통해서만 데이터를 주고받는다. product_id별 원본 정보(product_attrs/store_info/
# image 경로)도 함께 저장해서, STEP6가 상품별로 대조 평가하고 STEP8이 원본 이미지를
# 찾아 카피를 합성할 수 있게 한다.
GENERATED_COPY_RESULTS_PATH = str(BASE_DIR / "generated_copy_results.json")
payload = {
    "products": {
        PRODUCT_ID: {
            "product_attrs": product_attrs,
            "store_info": store_info,
            "image": PRODUCT_IMAGE_PATH,
        }
    },
    "results": results,
}
with open(GENERATED_COPY_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"\n생성된 카피 결과 저장 완료 -> {GENERATED_COPY_RESULTS_PATH}")
print("정량 평가는 `python step6_evaluate_generated_copy.py` 를 실행하세요.")
