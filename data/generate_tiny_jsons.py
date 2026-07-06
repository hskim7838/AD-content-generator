# -*- coding: utf-8 -*-
"""
generate_tiny_jsons.py
------------------------
파이프라인 1단계("이미지 -> tiny.json") 결과물의 샘플/placeholder 데이터를
개별 .json 파일로 저장하는 스크립트.

실제 서비스에서는 이 데이터가 "누끼 이미지를 LLM에 넣어 캡션/속성을 뽑는" 단계에서
자동 생성되지만, 이 파일은 카피라이팅 비교(compare_copy_llms.py, eval_copy.py) 단계를
독립적으로 테스트할 수 있도록 예시 tiny.json 5종을 만들어주는 용도입니다.

사용법
------
    python data/generate_tiny_jsons.py
    # -> ./tiny_jsons/ 폴더에 5개의 .json 파일 생성

    python data/generate_tiny_jsons.py --output-dir ./my_jsons --download
    # -> 코랩 환경이면 생성 직후 브라우저로 다운로드까지 트리거
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

# ----------------------------------------------------------------------------
# 샘플 데이터 (실제 서비스에서는 1단계 LLM 캡셔닝 결과로 대체됩니다)
# ----------------------------------------------------------------------------
TINY_DATA_LIST: List[Dict[str, str]] = [
    {
        "id": "bakery_product1",
        "image": "/home/ai3/AD-content/CAIG/tiny_dataset/images_test/bakery_product1.png",
        "new_caption": (
            "Assorted cafe bakery plate with golden croissant, chocolate-drizzled "
            "pastry, chip cookie, and dark iced coffee in a clear glass—warm premium "
            "treat set for breakfast or dessert."
        ),
    },
    {
        "id": "japanese_restaurant_sushi",
        "image": "/home/ai3/AD-content/CAIG/tiny_dataset/images_test/sushi_set.png",
        "new_caption": (
            "Premium assorted sushi platter featuring glossy salmon, tuna, and shrimp "
            "nigiri, served on a traditional wooden plate with a side of soy sauce—"
            "fresh, vibrant, and appetizing dine-in lunch specialized menu."
        ),
    },
    {
        "id": "flower_shop_bouquet",
        "image": "/home/ai3/AD-content/CAIG/tiny_dataset/images_test/flower_bouquet.png",
        "new_caption": (
            "Elegant seasonal flower bouquet wrapped in soft pastel pink paper, "
            "featuring vibrant red roses, white carnations, and eucalyptus leaves—"
            "romantic, high-quality gift set for anniversaries and celebrations."
        ),
    },
    {
        "id": "side_dish_kimchi",
        "image": "/home/ai3/AD-content/CAIG/tiny_dataset/images_test/fresh_kimchi.png",
        "new_caption": (
            "Authentic, freshly made Korean cabbage kimchi coated in rich red pepper "
            "paste, neatly packed in a clean transparent glass jar—homemade traditional "
            "side dish, vibrant texture, ready for local delivery."
        ),
    },
    {
        "id": "clothing_boutique_dress",
        "image": "/home/ai3/AD-content/CAIG/tiny_dataset/images_test/summer_dress.png",
        "new_caption": (
            "Lightweight linen summer midi dress in beige, hanging on a minimalist "
            "wooden hanger against a clean wall—casual, breathable, stylish daily wear "
            "suitable for online fashion lookbooks."
        ),
    },
]


def save_tiny_jsons(output_dir: str = "./tiny_jsons", trigger_download: bool = False) -> List[str]:
    """
    TINY_DATA_LIST를 개별 .json 파일로 저장한다.

    output_dir        : 저장할 폴더 (없으면 생성)
    trigger_download   : True면 코랩 환경에서 각 파일을 브라우저로 다운로드까지 트리거.
                          코랩이 아닌 환경(로컬/주피터랩)에서는 자동으로 무시됨.
    return             : 저장된 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for data in TINY_DATA_LIST:
        file_path = os.path.join(output_dir, f"{data['id']}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        saved_paths.append(file_path)
        print(f"저장 완료: {file_path}")

        if trigger_download:
            try:
                from google.colab import files  # 코랩이 아니면 ImportError
                files.download(file_path)
            except ImportError:
                pass  # 로컬/주피터랩 환경이면 조용히 건너뜀

    if trigger_download:
        print("\n[안내] 브라우저 팝업 차단이 켜져 있으면 파일이 1개만 다운로드될 수 있습니다.")
        print("차단 알림이 뜬다면 '팝업 항상 허용'을 누르고 다시 실행해 주세요.")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tiny.json 샘플 데이터 생성")
    parser.add_argument("--output-dir", default="./tiny_jsons", help="저장할 폴더 경로")
    parser.add_argument("--download", action="store_true",
                         help="코랩 환경에서 생성 직후 브라우저로 다운로드까지 트리거")
    args = parser.parse_args()

    save_tiny_jsons(output_dir=args.output_dir, trigger_download=args.download)
