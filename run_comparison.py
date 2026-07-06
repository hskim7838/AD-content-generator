# -*- coding: utf-8 -*-
"""
run_comparison.py
--------------------
전체 흐름(tiny.json 로드 -> 카피 생성 비교 -> 정량 평가) 실행 예시.

로컬/주피터랩:
    python examples/run_comparison.py --tiny-json data/tiny_jsons/flower_shop_bouquet.json --tone 미니멀

코랩:
    (레포를 clone/Drive 마운트한 뒤 아래처럼 함수만 가져다 쓰는 걸 권장)
    from compare_copy_llms import compare_models
    from eval_copy import evaluate_dataframe, show_full, print_comments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 레포 루트를 path에 추가 (examples/ 하위에서 실행해도 compare_copy_llms.py를 찾도록)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from compare_copy_llms import compare_models  # noqa: E402
from eval_copy import evaluate_dataframe, show_full, print_comments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="카피라이팅 LLM 비교 + 정량 평가 실행")
    parser.add_argument("--tiny-json", required=True, help="1단계 결과 tiny.json 경로")
    parser.add_argument("--tone", default="친근한",
                         choices=["미니멀", "강조형", "친근한", "고급스러운", "위트있는"])
    parser.add_argument("--store-name", default="샘플 매장")
    parser.add_argument("--store-phone", default="02-123-4567")
    parser.add_argument("--store-price", default="12,000원")
    parser.add_argument("--skip-semantic", action="store_true", help="임베딩 유사도 평가 생략(비용 절감)")
    parser.add_argument("--skip-judge", action="store_true", help="LLM judge 평가 생략(비용 절감)")
    args = parser.parse_args()

    with open(args.tiny_json, encoding="utf-8") as f:
        product_attrs = json.load(f)

    store_info = {
        "name": args.store_name,
        "phone": args.store_phone,
        "price": args.store_price,
    }

    base_df = compare_models(product_attrs, store_info, tone=args.tone)
    result = evaluate_dataframe(
        base_df, product_attrs, store_info, tone=args.tone,
        run_semantic=not args.skip_semantic,
        run_judge=not args.skip_judge,
    )

    print(show_full(result).to_string(index=False))
    print()
    print_comments(result)


if __name__ == "__main__":
    main()
