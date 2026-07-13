# -*- coding: utf-8 -*-
"""
compare_copy_llms.py
----------------------
"광고 문구(카피라이팅) 생성" 용도로 GPT 계열 모델에 같은 입력(product_attrs + store_info)을
실제로 돌려서 카피(JSON)를 생성하는 모듈. (copywriting_3.py에 주석으로 남아있던 원본 로직을
그대로 실행 가능한 형태로 복원한 것 — 로직 변경 없음)

[옵션]
- product / brand / sales       : 카피 생성 프롬프트에 실제 반영되는 상품/브랜드/판매 정보
- focusing_degree (0.0~1.0)     : 제품 강조(포커스) 정도. 높을수록 제품이 선명/부각됨
- background (텍스트)            : 원하는 배경 컨셉 설명
- background_intensity (0.0~1.0): 배경의 존재감/복잡도. 낮으면 심플·단색에 가까움
- preservation_degree (0.0~1.0) : 원본 이미지 보존 정도. 높을수록 원본 형태를 최대한 유지

focusing_degree / background_intensity / preservation_degree 는 카피 생성 자체보다는
"이후 ControlNet/SD 이미지 생성 단계"에서 쓰일 파라미터입니다.
to_image_gen_params()로 실제 SD/ControlNet 파라미터로 변환할 수 있습니다.

사용하려면 OPENAI_API_KEY 환경변수(또는 .env)가 설정되어 있어야 합니다.
"""

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
