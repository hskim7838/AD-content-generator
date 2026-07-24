"""
카피라이팅 + 해시태그 생성만 빠르게 써보는 예제.

이미지 생성(torch/diffusers/GPU)은 필요 없고,
requirements-text.txt 만 설치하면 바로 실행할 수 있습니다.

실행:
    pip install -r requirements-text.txt
    cp .env.example .env   # OPENAI_API_KEY 채워넣기
    python examples/quickstart_copy_and_hashtags.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 이 스크립트를 examples/ 밖(레포 루트)에서 실행하지 않아도 되도록
# adcg 패키지 경로를 sys.path에 추가합니다.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from adcg.config import TONE_OPTIONS  # noqa: E402
from adcg.prompting import generate_ad_copy, generate_ad_hashtags  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="상품/매장 정보로 광고 카피와 해시태그를 생성합니다."
    )
    parser.add_argument(
        "--info",
        default=str(REPO_ROOT / "examples" / "sample_product_info.json"),
        help="product_info.json 경로",
    )
    parser.add_argument(
        "--tone",
        choices=TONE_OPTIONS,
        default=None,
        help="문구 톤. 생략하면 info 파일의 tone 값을 사용합니다.",
    )
    parser.add_argument(
        "--copy-length",
        choices=("짧게", "보통", "긴"),
        default="보통",
        help="문구/해시태그 분량 (UI의 '글 길이' 옵션과 동일).",
    )
    parser.add_argument(
        "--copy-count",
        type=int,
        default=1,
        help="생성할 카피 개수.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
        help="사용할 모델명.",
    )
    parser.add_argument(
        "--background-prompt",
        default="",
        help=(
            "실제 파이프라인에서는 02_prompt/ad_prompt.json의 "
            "generation_prompt.background_prompt 값이 여기에 들어갑니다. "
            "이 데모에서는 생략 가능합니다."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    product_info = json.loads(
        Path(args.info).read_text(encoding="utf-8")
    )

    tone = args.tone or product_info.get("tone")

    print(f"[상품] {product_info.get('product_name')} / [매장] {product_info.get('store_name')}")
    print(f"[tone] {tone} / [글 길이] {args.copy_length}")
    print("-" * 50)

    copies = [
        generate_ad_copy(
            product_info={**product_info, "tone": tone},
            background_prompt=args.background_prompt,
            model=args.model,
        )
        for _ in range(args.copy_count)
    ]

    print("[카피]")
    print(json.dumps(copies, ensure_ascii=False, indent=2))

    hashtags = generate_ad_hashtags(
        product_info=product_info,
        background_prompt=args.background_prompt,
        tone=tone,
        copy_length=args.copy_length,
        model=args.model,
    )

    print("\n[해시태그]")
    print(json.dumps(hashtags, ensure_ascii=False, indent=2))

    output_path = REPO_ROOT / "outputs" / "quickstart_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"copies": copies, "hashtags": hashtags},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
