# -*- coding: utf-8 -*-
"""
compare_copy_llms.py
----------------------
"광고 문구(카피라이팅) 생성" 용도로 여러 텍스트 LLM을 같은 입력(tiny.json + 매장 정보)에
돌려보고 결과를 나란히 비교하는 주피터랩용 모듈.

비교 대상 (기본 구성)
  - OpenAI GPT-4.1 / GPT-5      : client.chat.completions.create (json_object 모드)
  - Naver CLOVA Studio (HCX-005): OpenAI 호환 엔드포인트라서 위와 동일한 방식으로 호출
  - Anthropic Claude            : anthropic SDK로 별도 호출

설계 방향
  - "카피라이팅 비교"가 목적이므로 이미지/비전 관련 코드는 전혀 포함하지 않음.
  - 각 LLM을 하나의 공통 인터페이스(BackendConfig)로 등록해두고,
    동일한 프롬프트 + 동일한 Pydantic 스키마로 호출 -> pandas DataFrame으로 비교.
  - 주피터랩 환경 기준이므로 Colab 전용 코드(!apt-get 등)는 제거하고,
    .env 파일 + python-dotenv로 API 키를 관리하는 방식을 사용.
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

# ----------------------------------------------------------------------------
# 0. 환경변수 로드 (.env 파일)
# ----------------------------------------------------------------------------
# 주피터랩 작업 폴더에 .env 파일을 만들고 아래처럼 채워두면 됩니다.
#
#   OPENAI_API_KEY=sk-...
#
load_dotenv()


# ----------------------------------------------------------------------------
# 1. 출력 스키마 (모든 모델이 동일한 스키마로 답하게 강제)
# ----------------------------------------------------------------------------
ToneType = Literal["미니멀", "강조형", "친근한", "고급스러운", "위트있는"]


class AdCopy(BaseModel):
    tone: ToneType = Field(description="이 카피가 지향하는 톤앤매너")
    title: str = Field(description="배너 메인 타이틀. 12자 내외")
    subtitle: str = Field(description="서브 카피. 20자 내외")
    price_text: Optional[str] = Field(default=None, description="가격 강조 문구. 없으면 null")
    cta: str = Field(description="행동 유도 문구(CTA)")


SYSTEM_PROMPT = """\
당신은 오프라인 소상공인을 위한 광고 카피라이터입니다.
- 과장 광고나 근거 없는 최상급 표현을 쓰지 않습니다.
- 타이틀은 12자 내외, 서브카피는 20자 내외로 간결하게 작성합니다.
- 주어진 상품/매장 정보에 없는 사실을 지어내지 않습니다.
- 반드시 아래 JSON 스키마 형식으로만 답하세요. 다른 설명은 절대 붙이지 마세요.

{schema}
"""


def _build_user_prompt(product_attrs: dict, store_info: dict, tone: str) -> str:
    return f"""\
[상품 속성 (tiny.json)]
{json.dumps(product_attrs, ensure_ascii=False, indent=2)}

[매장 정보]
{json.dumps(store_info, ensure_ascii=False, indent=2)}

[요청]
톤앤매너 "{tone}" 에 맞는 광고 카피 1세트를 생성해 주세요.
"""


# ----------------------------------------------------------------------------
# 2. 백엔드 설정 (비교하고 싶은 모델을 여기에 등록)
# ----------------------------------------------------------------------------
@dataclass
class BackendConfig:
    name: str                        # 비교 결과 표에 표시될 이름
    provider: Literal["openai_compatible"]
    model: str
    base_url: Optional[str] = None   # 기본 OpenAI 엔드포인트가 아닌 경우에만 지정
    api_key_env: str = "OPENAI_API_KEY"


# 필요에 따라 자유롭게 추가/삭제하세요.
BACKENDS: List[BackendConfig] = [
    BackendConfig(
        name="GPT-5.5",
        provider="openai_compatible",
        model="gpt-5.5",
        base_url=None,  # 기본 OpenAI 엔드포인트
        api_key_env="OPENAI_API_KEY",
    ),
    BackendConfig(
        name="GPT-5.4 mini",
        provider="openai_compatible",
        model="gpt-5.4-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    ),
    BackendConfig(
        name="GPT-5.4 nano",
        provider="openai_compatible",
        model="gpt-5.4-nano",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
    ),
]


# ----------------------------------------------------------------------------
# 3. 호출 함수 (provider별로 분기, 결과는 항상 raw text 반환)
# ----------------------------------------------------------------------------
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
    """모델이 ```json ... ``` 코드펜스로 감싸서 답하는 경우까지 방어적으로 처리."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ----------------------------------------------------------------------------
# 4. 비교 실행 함수
# ----------------------------------------------------------------------------
def compare_models(
    product_attrs: dict,
    store_info: dict,
    tone: str = "친근한",
    backends: Optional[List[BackendConfig]] = None,
    max_retries: int = 1,
) -> pd.DataFrame:
    """
    등록된 모든 백엔드에 동일한 입력을 보내고 결과를 pandas DataFrame으로 비교.
    실패한 모델은 error 컬럼에 사유를 남기고 계속 진행한다 (한 모델의 실패가
    전체 비교를 막지 않도록).
    """
    backends = backends or BACKENDS
    schema_hint = json.dumps(AdCopy.model_json_schema(), ensure_ascii=False, indent=2)
    system = SYSTEM_PROMPT.format(schema=schema_hint)
    user = _build_user_prompt(product_attrs, store_info, tone)

    rows = []
    for cfg in backends:
        row = {"model": cfg.name, "tone": tone, "title": None, "subtitle": None,
               "price": None, "cta": None, "latency_sec": None, "error": None}
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
            except (json.JSONDecodeError, ValidationError, RuntimeError, Exception) as e:  # noqa: BLE001
                last_err = e
                continue
        if last_err is not None:
            row["error"] = f"{type(last_err).__name__}: {last_err}"
            row["latency_sec"] = round(time.time() - start, 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. 사용 예시 (주피터랩 셀에 그대로 붙여넣어도 됨)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1번 단계 LLM이 만들어 둔 tiny.json을 그대로 불러온다.
    # (경로는 실제 저장 위치에 맞게 수정하세요. 예: "./outputs/tiny.json")
    with open("tiny.json", encoding="utf-8") as f:
        product_attrs = json.load(f)

    # 매장 정보는 사용자가 입력 폼에서 넣은 값이라고 가정 (필요 시 별도 json으로 관리 가능)
    store_info = {
        "name": "옥동정육식당",
        "phone": "02-123-4567",
        "price": "12,000원",
    }

    df = compare_models(product_attrs, store_info, tone="친근한")
    # 주피터랩 셀에서는 마지막 줄에 df 만 남겨두면 표 형태로 예쁘게 렌더링됩니다.
    print(df.to_string(index=False))
