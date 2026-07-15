# -*- coding: utf-8 -*-
"""copywriting_4 (VSCode/로컬 실행용, 카피 문구 생성 전용 버전)

이미지 합성 없이 카피 텍스트만 생성/출력하는 로컬 파이프라인.

사전 준비:
  1) pip install openai pandas python-dotenv
  2) 프로젝트 루트에 .env 파일을 만들고 아래처럼 API 키를 넣어두세요.
       OPENAI_API_KEY=sk-...
     (.env가 없다면 실행 전에 직접 os.environ["OPENAI_API_KEY"] = "..." 로 설정해도 됩니다.)

전체 파이프라인:
  STEP 1. 환경 설정 (OpenAI 클라이언트, 매장 정보)
  STEP 2. 이미지 폴더 -> tiny.json 자동 생성 (GPT-4o Vision 캡션)
  STEP 3. tiny.json 통합 -> option_pools.json / run_config.json 생성
  STEP 4. compare_copy_llms.py 모듈 파일 생성 (카피 생성 함수)
  STEP 5. 랜덤 조합 9개 -> GPT 실제 호출 -> 카피 문구(title/subtitle/price/cta)만 출력
"""

# ══════════════════════════════════════════════════════════════
# STEP 1. 환경 설정
# ══════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import re
import shutil
import base64

from dotenv import load_dotenv
load_dotenv()  # 프로젝트 루트의 .env에서 OPENAI_API_KEY를 읽어옴

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY=sk-... 를 추가하거나 "
        "os.environ['OPENAI_API_KEY']를 직접 설정한 뒤 다시 실행하세요."
    )

from openai import OpenAI
client = OpenAI()

# 여기 3곳만 본인 환경/매장에 맞게 수정하세요 ─────────────────
INPUT_DIR = "/Users/apple/Desktop/AD-content-generator/bakery.png"  # 원본 이미지들이 들어있는 폴더

STORE_INFO = {
    # 매장 하나로 고정, 아래 폴더 안 모든 이미지(상품)에 공통 적용됩니다.
    "store_type": "",   # 예: "베이커리 카페"
    "store_name": "",   # 예: "OO베이커리"
    "location": "",     # 예: "서울 마포구"
    "highlight": "",    # 예: "매장 강조 문구" (없으면 빈 문자열로 둬도 됨)
    "phone": "",
    "price": "",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

BASE_DIR = "./tiny_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "jsons")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 2. 이미지 폴더 -> tiny.json 자동 생성 (GPT-4o Vision으로 캡션 자동 생성)
# --------------------------------------------------------------
# - INPUT_DIR 폴더 안의 이미지를 전부 스캔해서, 이미지마다:
#     1) GPT-4o Vision이 이미지를 보고 "상품 설명 — 배경/분위기 설명" 캡션을 자동 생성
#     2) 파일명에서 id를 자동으로 만들어 tiny.json(=id.json)으로 저장
# - store_info(매장 정보)는 STORE_INFO 하나로 고정해서 모든 이미지에 공통 적용합니다.
# - 폴더 구조:
#     ./tiny_dataset/
#         ├── images/     <- 원본 이미지 복사본
#         └── jsons/      <- id.json (image 필드는 로컬 경로로 갱신됨)
#
# ※ GPT-4o가 만든 캡션은 어디까지나 초안입니다. 결과가 이상하면
#    생성된 jsons/*.json의 new_caption만 사람이 한 번 검수/수정하세요.
# ══════════════════════════════════════════════════════════════
def make_id_from_filename(image_path: str) -> str:
    """파일명에서 tiny.json의 id를 자동 생성 (예: bakery.png -> 'bakery')."""
    stem = os.path.splitext(os.path.basename(image_path))[0].strip().lower()
    stem = re.sub(r"[^0-9a-z가-힣]+", "_", stem)
    return stem.strip("_") or "item"


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption_with_gpt4o(image_path: str, model: str = "gpt-4o") -> str:
    """
    이미지를 GPT-4o Vision에 직접 보여주고 "상품 설명 — 배경/분위기 설명"
    형식의 영문 캡션을 자동 생성한다. (사람이 new_caption을 직접 타이핑하지 않아도 됨)
    """
    b64 = _encode_image_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "png" if ext == "png" else "jpeg"

    prompt = (
        "Look at this product photo and write ONE English caption in exactly this format:\n"
        "\"<short product description> — <background/mood description>\"\n"
        "Use an em dash (—) to separate the two halves. "
        "Product description: plainly list the food/drink/product items visible. "
        "Background description: describe the setting, lighting, mood. "
        "Return ONLY the caption text, nothing else."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
            ],
        }],
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def list_input_images(input_dir: str):
    # INPUT_DIR이 폴더가 아니라 이미지 파일 1개를 직접 가리켜도 동작하도록 지원
    if os.path.isfile(input_dir):
        if input_dir.lower().endswith(IMAGE_EXTS):
            return [input_dir]
        print(f"⚠ 이미지 파일이 아닙니다: {input_dir}")
        return []

    if not os.path.isdir(input_dir):
        print(f"⚠ 입력 폴더/파일을 찾을 수 없습니다: {input_dir}")
        print("   -> INPUT_DIR 값을 실제 이미지 파일 경로 또는 이미지들이 들어있는 폴더 경로로 수정해주세요.")
        return []

    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )


def save_image_locally(src_path: str, item_id: str) -> str:
    """원본 이미지를 IMG_DIR로 복사하고, 새 로컬 경로를 반환"""
    ext = os.path.splitext(src_path)[1] or ".png"
    dst_path = os.path.join(IMG_DIR, f"{item_id}{ext}")
    shutil.copy(src_path, dst_path)
    return dst_path


print("\n=== [STEP 2] 이미지 스캔 + 자동 캡션 생성 + JSON 저장 시작 ===")
generated_json_paths = []
for src_path in list_input_images(INPUT_DIR):
    item_id = make_id_from_filename(src_path)
    print(f"\n[{item_id}] 처리 중: {src_path}")

    try:
        caption = generate_caption_with_gpt4o(src_path)
    except Exception as e:
        print(f"  ❌ 캡션 자동 생성 실패: {e} -> 이 이미지는 건너뜁니다.")
        continue
    print(f"  📝 자동 생성 캡션: {caption}")

    local_img_path = save_image_locally(src_path, item_id)
    print(f"  ✅ 이미지 저장 완료 -> {local_img_path}")

    tiny_data = {
        "id": item_id,
        "image": local_img_path,
        "new_caption": caption,
        "store_info": STORE_INFO,
    }
    json_path = os.path.join(JSON_DIR, f"{item_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tiny_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ json 저장 완료 -> {json_path}")
    generated_json_paths.append(json_path)

print("\n=== [STEP 2] 전체 저장 완료 ===")
print(f"이미지: {IMG_DIR}")
print(f"JSON  : {JSON_DIR}")
print(f"생성된 tiny.json 개수: {len(generated_json_paths)}")


# ══════════════════════════════════════════════════════════════
# STEP 3. tiny.json 통합 -> option_pools.json / run_config.json 자동 생성
# ══════════════════════════════════════════════════════════════
def _split_caption(caption: str) -> tuple[str, str]:
    """
    new_caption을 "상품 설명 — 배경/분위기 설명" 구조로 가정하고 둘로 분리.
    (캡션에 em-dash로 상품/배경을 구분해두는 패턴을 활용)
    구분자가 없으면 전체를 상품 설명으로 보고 배경은 빈 문자열로 반환.
    """
    if not caption:
        return "", ""
    for dash in ("—", "–", " - "):
        if dash in caption:
            product_part, _, background_part = caption.partition(dash)
            return product_part.strip().rstrip(","), background_part.strip().rstrip(".")
    return caption.strip(), ""


def _shorten_text(text: str, max_words: int = 8) -> str:
    """긴 캡션 문장을 앞부분 몇 단어만 남기고 축약 (product 필드용 초안)."""
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def build_option_pools_from_tiny_jsons(
    json_dir: str,
    option_pools_path: str = "./option_pools.json",
    run_config_path: str = "./run_config.json",
    tone: str = "친근한",
    run_config_source_id: str | None = None,
) -> tuple[dict, dict]:
    """
    json_dir 안의 모든 tiny.json(id.json)을 읽어서
    option_pools.json(매장/옵션 풀)과 run_config.json(이번 요청 옵션)을 자동 생성한다.

    - store_info : STORE_INFO(고정, 공통) 기반으로 채운다.
    - products   : json_dir 안 모든 이미지에서 자동 추출한 product 목록 (폴더에 이미지가
                   여러 장이면 products 풀도 자동으로 여러 개가 된다).
    - run_config : run_config_source_id로 지정한 상품(없으면 첫 번째 상품) 기준으로 생성.
    - 자동 추정값은 어디까지나 '초안'이므로, 실제 서비스에서는 생성된 JSON을
      그대로 쓰지 말고 사람이 한 번 검수/수정하는 것을 권장한다.
    """
    json_paths = sorted(
        os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")
    )
    if not json_paths:
        raise FileNotFoundError(f"{json_dir} 안에 tiny.json이 없습니다. 먼저 STEP 2를 실행하세요.")

    products, tiny_data_by_id = [], {}
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        product_guess, _ = _split_caption(data.get("new_caption", ""))
        product_guess = _shorten_text(product_guess, max_words=8) or data["id"]
        products.append(product_guess)
        tiny_data_by_id[data["id"]] = data

    store_name = STORE_INFO.get("store_name") or "매장명 미입력"
    highlight = STORE_INFO.get("highlight") or None
    store_info = {
        "name": store_name,
        "phone": STORE_INFO.get("phone") or "문의 필요",
        "price": STORE_INFO.get("price") or "가격 문의",
    }

    pools = {
        "store_info": store_info,
        "tones": ["미니멀", "강조형", "친근한", "고급스러운", "위트있는"],
        "products": products,
        "brands": [store_name],
        "sales": [highlight] if highlight else [None],
    }

    run_source_id = run_config_source_id or next(iter(tiny_data_by_id))
    run_source_data = tiny_data_by_id[run_source_id]
    product_guess, background_guess = _split_caption(run_source_data.get("new_caption", ""))
    product_guess = _shorten_text(product_guess, max_words=8) or run_source_id

    run_cfg = {
        "product_attrs_path": os.path.join(json_dir, f"{run_source_id}.json"),
        "image_path": run_source_data.get("image"),
        "tone": tone,
        "product": product_guess,
        "brand": store_name,
        "sales": highlight,
        "focusing_degree": 0.7,
        "background": background_guess or None,
        "background_intensity": 0.3,
        "preservation_degree": 0.6,
    }

    with open(option_pools_path, "w", encoding="utf-8") as f:
        json.dump(pools, f, ensure_ascii=False, indent=2)
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(f"✅ 자동 생성 완료 -> {option_pools_path}, {run_config_path}")
    print(f"   (product {len(products)}개 자동 추출, run_config 기준 상품: {run_source_id})")
    print("   (자동 추정된 product/background 값은 검수 후 필요시 수정하세요)")
    return pools, run_cfg


OPTION_POOLS_PATH = "./option_pools.json"
RUN_CONFIG_PATH = "./run_config.json"

print("\n=== [STEP 3] option_pools.json / run_config.json 생성 ===")
if generated_json_paths or not (os.path.exists(OPTION_POOLS_PATH) and os.path.exists(RUN_CONFIG_PATH)):
    build_option_pools_from_tiny_jsons(
        JSON_DIR,
        option_pools_path=OPTION_POOLS_PATH,
        run_config_path=RUN_CONFIG_PATH,
    )
else:
    print("⚠ 새로 생성된 tiny.json이 없고 기존 option_pools.json/run_config.json이 있어 재사용합니다.")


# ══════════════════════════════════════════════════════════════
# STEP 4. compare_copy_llms.py 모듈 파일 생성
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

with open("compare_copy_llms.py", "w", encoding="utf-8") as f:
    f.write(_COMPARE_COPY_LLMS_SOURCE)

print("\n=== [STEP 4] compare_copy_llms.py 모듈 파일 생성 완료 ===")

from compare_copy_llms import compare_models, BackendConfig


# ══════════════════════════════════════════════════════════════
# STEP 5. 랜덤 조합 9개 -> compare_models() 실제 GPT 호출 -> 카피 문구만 출력
# --------------------------------------------------------------
# - option_pools.json에서 tone/product/brand/sales 등을 랜덤 조합해 9개 생성
# - 조합 하나당 모델 1개씩만 호출하되, 3개 모델(GPT-5.5 / mini / nano)이
#   번갈아가며(라운드로빈) 골고루 섞이게 함
# - 이미지 합성 없이 GPT가 만든 title/subtitle/price/cta 텍스트만 출력
# ══════════════════════════════════════════════════════════════
import random

random.seed(42)

with open(RUN_CONFIG_PATH, encoding="utf-8") as f:
    _run_cfg_for_random = json.load(f)
PRODUCT_ATTRS_PATH = _run_cfg_for_random["product_attrs_path"]

with open(PRODUCT_ATTRS_PATH, encoding="utf-8") as f:
    product_attrs = json.load(f)

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
print("\n=== [STEP 5] 랜덤 조합 9개 ===")
for i, c in enumerate(combos, 1):
    print(f"[{i}] {c}")

print("\n=== [STEP 5] 9개 조합 실제 GPT 호출 (카피 문구만 출력) 시작 ===")
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

    results.append({**combo, **copy})

print("\n=== [STEP 5] 전체 완료 ===")
print(f"성공: {len(results)} / 9")

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

# ── 정량 평가(STEP 6, 별도 파일)로 넘겨주기 위해 결과를 JSON으로 저장 ──
# STEP 6(step6_evaluate_generated_copy.py)은 이 파일과 별개의 프로세스로 실행되므로,
# 파이썬 변수(results/product_attrs/store_info)를 공유하지 않고 파일을 통해 전달한다.
GENERATED_COPY_RESULTS_PATH = "./generated_copy_results.json"
with open(GENERATED_COPY_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n생성된 카피 결과 저장 완료 -> {GENERATED_COPY_RESULTS_PATH}")
print("정량 평가는 별도로 `python step6_evaluate_generated_copy.py` 를 실행하세요.")
