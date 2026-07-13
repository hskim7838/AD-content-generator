# -*- coding: utf-8 -*-
"""
step4_config_based_run.py
----------------------------
[4단계] run_config.json + option_pools.json 을 읽어 compare_models() 실행.
파일 내용만 바꾸면 코드 수정 없이 다른 톤/상품/옵션으로 재실행 가능.

경로는 config.py에서 가져옵니다.
"""

import json
import os

from dotenv import load_dotenv

import config
from compare_copy_llms import compare_models, to_image_gen_params
from data_prep import build_configs_from_tiny_json

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


def run_compare_models_from_config(
    config_path: str = config.RUN_CONFIG_PATH,
    option_pools_path: str = config.OPTION_POOLS_PATH,
):
    """run_config.json(요청별 옵션) + option_pools.json(매장 정보)을 읽어 compare_models()를 실행."""
    with open(config_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    with open(option_pools_path, "r", encoding="utf-8") as f:
        store_info = json.load(f)["store_info"]

    with open(run_cfg["product_attrs_path"], "r", encoding="utf-8") as f:
        product_attrs = json.load(f)

    return compare_models(
        product_attrs,
        store_info,
        tone=run_cfg.get("tone", "친근한"),
        product=run_cfg.get("product"),
        brand=run_cfg.get("brand"),
        sales=run_cfg.get("sales"),
        focusing_degree=run_cfg.get("focusing_degree", 0.5),
        background=run_cfg.get("background"),
        background_intensity=run_cfg.get("background_intensity", 0.5),
        preservation_degree=run_cfg.get("preservation_degree", 0.5),
    )


if __name__ == "__main__":
    if not os.path.exists(config.PRODUCT_ATTRS_PATH):
        raise FileNotFoundError(
            f"{config.PRODUCT_ATTRS_PATH} 가 없습니다. step1_prepare_tiny_json.py를 먼저 실행하세요."
        )

    # run_config.json / option_pools.json이 아직 없으면 tiny.json으로부터 자동 생성
    # (overwrite=False -> 이미 파일이 있으면 직접 수정해둔 값을 덮어쓰지 않음)
    build_configs_from_tiny_json(
        config.PRODUCT_ATTRS_PATH,
        option_pools_path=config.OPTION_POOLS_PATH,
        run_config_path=config.RUN_CONFIG_PATH,
        overwrite=False,
    )

    df = run_compare_models_from_config()
    print(df)

    with open(config.RUN_CONFIG_PATH, encoding="utf-8") as f:
        _cfg_preview = json.load(f)

    preview = to_image_gen_params(
        focusing_degree=_cfg_preview.get("focusing_degree", 0.5),
        background_intensity=_cfg_preview.get("background_intensity", 0.5),
        preservation_degree=_cfg_preview.get("preservation_degree", 0.5),
    )
    print("이미지 생성 파라미터 미리보기:", preview)
