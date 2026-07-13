# -*- coding: utf-8 -*-
"""
step1_prepare_tiny_json.py
----------------------------
[1단계] 상품 사진 + 캡션 + 매장정보 -> ./tiny_dataset/ 에 정리 저장

경로는 이제 config.py 한 곳에서만 관리합니다.
이미지 경로를 바꾸고 싶으면 config.py의 PRODUCT_IMAGE_PATH만 수정하세요.
"""

import config
from data_prep import save_tiny_dataset

tiny_data_list = [
    {
        "id": config.PRODUCT_ID,
        "image": config.PRODUCT_IMAGE_PATH,
        "new_caption": config.PRODUCT_CAPTION,
        "store_info": config.STORE_INFO_RAW,
    },
]

if __name__ == "__main__":
    save_tiny_dataset(tiny_data_list, img_dir=config.IMG_DIR, json_dir=config.JSON_DIR)
