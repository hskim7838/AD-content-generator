# -*- coding: utf-8 -*-
"""
data_prep.py
-------------
- 원본 이미지 + 캡션 정보를 ./tiny_dataset/ 아래에 정리해서 저장
- tiny.json(상품 속성 파일) 한 개로부터 option_pools.json / run_config.json 자동 생성
- option_pools.json 로드 + 랜덤 조합 생성

Colab 버전과 다른 점: google.colab.drive / google.colab.files 의존성을 모두 제거했고,
모든 경로는 로컬 상대 경로 기준으로 동작합니다.
"""

from __future__ import annotations

import json
import os
import random
import shutil

BASE_DIR = "./tiny_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
JSON_DIR = os.path.join(BASE_DIR, "jsons")


def _split_caption(caption: str) -> tuple[str, str]:
    """
    new_caption을 "상품 설명 — 배경/분위기 설명" 구조로 가정하고 둘로 분리.
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


def save_image_locally(data: dict, img_dir: str = IMG_DIR) -> str | None:
    """image 필드의 원본 경로를 img_dir로 복사하고, 새 로컬 경로를 반환"""
    os.makedirs(img_dir, exist_ok=True)
    src = data["image"]
    ext = os.path.splitext(src)[1] or ".png"
    dst_path = os.path.join(img_dir, f"{data['id']}{ext}")

    if not os.path.exists(src):
        print(f"  ⚠ [{data['id']}] 이미지 경로를 찾을 수 없습니다: {src}")
        print("     -> tiny_data_list의 'image' 값을 실제 파일 경로로 수정해주세요.")
        return None

    shutil.copy(src, dst_path)
    return dst_path


def save_tiny_dataset(tiny_data_list: list[dict], img_dir: str = IMG_DIR, json_dir: str = JSON_DIR) -> None:
    """
    tiny_data_list(이미지 경로 + 캡션 + 매장정보 리스트)를 받아
    이미지는 img_dir로 복사하고, json은 json_dir에 저장한다.
    """
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    print("\n=== 이미지 + JSON 개별 저장 시작 ===")
    for data in tiny_data_list:
        local_img_path = save_image_locally(data, img_dir=img_dir)
        if not local_img_path:
            continue
        print(f"  ✅ [{data['id']}] 이미지 저장 완료 -> {local_img_path}")

        data_to_save = dict(data)
        data_to_save["image"] = local_img_path
        json_path = os.path.join(json_dir, f"{data['id']}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"  ✅ [{data['id']}] json 저장 완료  -> {json_path}")

    print("=== 전체 저장 완료 ===")
    print(f"이미지: {img_dir}")
    print(f"JSON  : {json_dir}")


def build_configs_from_tiny_json(
    tiny_json_path: str,
    option_pools_path: str = "./option_pools.json",
    run_config_path: str = "./run_config.json",
    tone: str = "친근한",
    overwrite: bool = True,
) -> tuple[dict, dict]:
    """
    tiny.json(상품 속성 파일) 하나를 읽어서
    option_pools.json(매장/옵션 풀)과 run_config.json(이번 요청 옵션)을 자동 생성한다.

    - store_info : tiny.json 안의 store_info(store_type/store_name/location/highlight)를
                   기반으로 채우되, 비어있는 값은 기본값으로 대체한다.
    - product/background : new_caption을 "상품 설명 — 배경 설명"으로 분리해 추정한다.
    - 자동 추정값은 어디까지나 '초안'이므로, 실제 서비스에서는 생성된 JSON을
      그대로 쓰지 말고 사람이 한 번 검수/수정하는 것을 권장한다.
    """
    with open(tiny_json_path, "r", encoding="utf-8") as f:
        tiny_data = json.load(f)

    raw_store = tiny_data.get("store_info", {}) or {}
    store_name = raw_store.get("store_name") or "매장명 미입력"
    store_type = raw_store.get("store_type") or "가게"
    highlight = raw_store.get("highlight") or None

    product_guess, background_guess = _split_caption(tiny_data.get("new_caption", ""))
    product_guess = _shorten_text(product_guess, max_words=8) or store_type

    store_info = {
        "name": store_name,
        "phone": raw_store.get("phone") or "문의 필요",
        "price": raw_store.get("price") or "가격 문의",
    }
    pools = {
        "store_info": store_info,
        "tones": ["미니멀", "강조형", "친근한", "고급스러운", "위트있는"],
        "products": [product_guess],
        "brands": [store_name],
        "sales": [highlight] if highlight else [None],
    }

    run_cfg = {
        "product_attrs_path": tiny_json_path,
        "tone": tone,
        "product": product_guess,
        "brand": store_name,
        "sales": highlight,
        "focusing_degree": 0.7,
        "background": background_guess or None,
        "background_intensity": 0.3,
        "preservation_degree": 0.6,
    }

    if overwrite or not os.path.exists(option_pools_path):
        with open(option_pools_path, "w", encoding="utf-8") as f:
            json.dump(pools, f, ensure_ascii=False, indent=2)

    if overwrite or not os.path.exists(run_config_path):
        with open(run_config_path, "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(f"✅ 자동 생성 완료 -> {option_pools_path}, {run_config_path}")
    print("   (자동 추정된 product/brand/background/sales 값은 검수 후 필요시 수정하세요)")
    return pools, run_cfg


def load_option_pools(json_path: str = "./option_pools.json") -> dict:
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


def generate_combos(n: int = 9, json_path: str = "./option_pools.json") -> list[dict]:
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
