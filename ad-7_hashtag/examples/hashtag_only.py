"""
상품 설명/매장 정보만으로 추천 키워드(해시태그)를 생성하는 코드입니다.
이미지 누끼(전처리), 배경 생성, 이미지 합성 등 adcg 전체 파이프라인은
전혀 거치지 않습니다. GPU도 필요 없고, OpenAI API 키만 있으면 됩니다.

세 가지 방식으로 실행할 수 있습니다.

1) 내장된 5개 예시 실행
    python examples/hashtag_only.py
    python examples/hashtag_only.py --example 3

2) CLI 인자로 임의의 상품 정보 직접 입력 (어떤 값이든, 일부만 넣어도 동작)
    python examples/hashtag_only.py \\
        --product-name "수제 캔들" --store-name "포근한 밤" \\
        --tone 감성적인 --copy-length 긴

    글자 길이를 프리셋(짧게/보통/긴) 대신 직접 숫자로 지정할 수도 있습니다.
    python examples/hashtag_only.py \\
        --product-name "수제 캔들" --hashtag-count 8 --max-chars 12

3) JSON 파일로 상품 정보 입력
    python examples/hashtag_only.py --info my_product.json --tone 위트있는

4) 대화형으로 값 입력 (터미널에서 하나씩 질문)
    python examples/hashtag_only.py --interactive

실행 전:
    pip install openai python-dotenv
    export OPENAI_API_KEY='sk-...'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from adcg.config import TONE_OPTIONS  # noqa: E402
from adcg.prompting import generate_ad_hashtags  # noqa: E402

# 문구(카피/해시태그) 생성에 사용할 모델
COPY_MODEL = "gpt-5.4-mini"

# product_info에 넣을 수 있는 필드 (전부 선택 사항 — 비워둬도 됩니다)
PRODUCT_INFO_FIELDS = (
    "product_name",
    "store_name",
    "store_type",
    "product_category",
    "product_description",
    "features",
    "target_customer",
    "promo_target",
    "promotion",
    "tone",
    "draft_copy",
)


def _run(title: str, product_info: dict, **kwargs) -> dict:
    print(f"\n=== {title} ===")

    result = generate_ad_hashtags(
        product_info=product_info,
        tone=kwargs.get("tone") or product_info.get("tone"),
        copy_length=kwargs.get("copy_length", "보통"),
        hashtag_count=kwargs.get("hashtag_count"),
        max_chars_per_tag=kwargs.get("max_chars_per_tag"),
        model=COPY_MODEL,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(" ".join(result["hashtags"]))
    return result


# ------------------------------------------------------------------
# 내장 예시 5개
# ------------------------------------------------------------------
def example_1_bakery() -> None:
    _run(
        "예시 1: 동네 베이커리 신메뉴",
        {
            "product_name": "수제 버터 크루아상",
            "store_name": "아침빵집",
            "store_type": "동네 베이커리",
            "product_category": "베이커리",
            "product_description": "매일 아침 직접 반죽해 굽는 프랑스산 버터 크루아상입니다.",
            "features": ["프랑스산 버터 100% 사용", "매일 아침 소량 생산"],
            "target_customer": "출근길 직장인",
            "promo_target": "재택 대신 출근을 시작한 20-30대 사회초년생",
            "promotion": "오픈 기념 아메리카노 500원 할인",
            "tone": "친근한",
            "draft_copy": "",
        },
        copy_length="보통",
    )


def example_2_home_living() -> None:
    _run(
        "예시 2: 인테리어/생활용품 (짧게)",
        {
            "product_name": "세탁기 받침대",
            "store_name": "홈리빙마켓",
            "store_type": "소상공인 매장",
            "product_category": "생활/인테리어",
            "product_description": "세탁기 아래에 놓아 층간소음과 진동을 줄여주는 받침대입니다.",
            "features": ["층간소음 저감", "미끄럼 방지", "간편 설치"],
            "target_customer": "아파트/빌라 거주 1인 가구",
            "promo_target": "신혼부부, 자취 시작하는 1인 가구",
            "promotion": "",
            "tone": "미니멀한",
            "draft_copy": "세탁기 밑에 놓기만 해도 층간소음이 줄어요",
        },
        copy_length="짧게",
    )


def example_3_leather_atelier() -> None:
    _run(
        "예시 3: 핸드메이드 가죽공방 (길게)",
        {
            "product_name": "핸드메이드 가죽 카드지갑",
            "store_name": "아뜰리에 소",
            "store_type": "가죽공방",
            "product_category": "패션잡화",
            "product_description": "이탈리아산 식물성 무두질 가죽으로 한 땀 한 땀 제작한 카드지갑입니다.",
            "features": ["이탈리아산 베지터블 가죽", "각인 서비스 제공"],
            "target_customer": "선물을 찾는 30-40대",
            "promo_target": "",
            "promotion": "",
            "tone": "고급스러운",
            "draft_copy": "",
        },
        copy_length="긴",
    )


def example_4_gift_shop() -> None:
    _run(
        "예시 4: 소품샵 (위트있는 톤)",
        {
            "product_name": "미니 무선 선풍기",
            "store_name": "쿨브리즈",
            "store_type": "온라인 소품샵",
            "product_category": "생활가전",
            "product_description": "휴대가 간편한 저소음 무선 미니 선풍기입니다.",
            "features": ["저소음 모터", "USB-C 고속 충전", "3단 풍속 조절"],
            "target_customer": "여름 캠핑족, 사무실 직장인",
            "promo_target": "",
            "promotion": "",
            "tone": "위트있는",
            "draft_copy": "",
        },
        copy_length="보통",
    )


def example_5_industrial_parts() -> None:
    _run(
        "예시 5: 산업용품 도매 (전문적인 톤)",
        {
            "product_name": "전동 지게차 부품",
            "store_name": "산업부품마트",
            "store_type": "산업용품 도매",
            "product_category": "산업/공구",
            "product_description": "내구성이 뛰어난 산업용 전동 지게차 교체 부품입니다.",
            "features": ["고내구성 소재", "표준 규격 호환"],
            "target_customer": "물류창고, 제조 공장, 건설 현장 관리자",
            "promo_target": "",
            "promotion": "",
            "tone": "전문적인",
            "draft_copy": "",
        },
        copy_length="보통",
    )


EXAMPLES = {
    1: example_1_bakery,
    2: example_2_home_living,
    3: example_3_leather_atelier,
    4: example_4_gift_shop,
    5: example_5_industrial_parts,
}


# ------------------------------------------------------------------
# 임의 입력 처리 (CLI 인자 / JSON 파일 / 대화형)
# ------------------------------------------------------------------
def _parse_features(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_info_from_args(args: argparse.Namespace) -> dict:
    return {
        "product_name": args.product_name or "",
        "store_name": args.store_name or "",
        "store_type": args.store_type or "",
        "product_category": args.product_category or "",
        "product_description": args.product_description or "",
        "features": _parse_features(args.features),
        "target_customer": args.target_customer or "",
        "promo_target": args.promo_target or "",
        "promotion": args.promotion or "",
        "tone": args.tone or "",
        "draft_copy": args.draft_copy or "",
    }


def _build_info_interactively() -> dict:
    print("상품/매장 정보를 입력하세요. 모르는 항목은 그냥 Enter로 넘어가도 됩니다.\n")

    def ask(label: str) -> str:
        return input(f"{label}: ").strip()

    info = {
        "product_name": ask("상품명"),
        "store_name": ask("매장명"),
        "store_type": ask("매장 유형"),
        "product_category": ask("상품 카테고리"),
        "product_description": ask("상품 설명"),
        "features": _parse_features(ask("주요 특징 (쉼표로 구분)")),
        "target_customer": ask("주 고객층"),
        "promo_target": ask("홍보 대상"),
        "promotion": ask("프로모션"),
        "draft_copy": ask("홍보 문구 초안"),
    }

    print(f"\n선택 가능한 tone: {', '.join(TONE_OPTIONS)} (다른 값도 입력 가능)")
    info["tone"] = ask("tone")

    return info


def _has_custom_input(args: argparse.Namespace) -> bool:
    custom_fields = (
        args.info,
        args.product_name,
        args.store_name,
        args.store_type,
        args.product_category,
        args.product_description,
        args.features,
        args.target_customer,
        args.promo_target,
        args.promotion,
        args.draft_copy,
    )
    return args.interactive or any(custom_fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="상품/매장 정보(무엇이든)로 추천 해시태그를 생성합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=sorted(EXAMPLES),
        default=None,
        help="내장된 예시 1~5 중 하나만 실행.",
    )
    parser.add_argument("--info", help="상품 정보 JSON 파일 경로.")
    parser.add_argument("--interactive", action="store_true", help="터미널에서 하나씩 물어보며 입력.")

    info_group = parser.add_argument_group("직접 입력 (전부 선택 사항)")
    info_group.add_argument("--product-name")
    info_group.add_argument("--store-name")
    info_group.add_argument("--store-type")
    info_group.add_argument("--product-category")
    info_group.add_argument("--product-description")
    info_group.add_argument("--features", help="쉼표로 구분한 특징 목록. 예: '방수, 초경량'")
    info_group.add_argument("--target-customer")
    info_group.add_argument("--promo-target", help="홍보 대상 (target-customer와 별개로 지정 가능)")
    info_group.add_argument("--promotion")
    info_group.add_argument("--draft-copy", help="홍보 문구 초안")

    gen_group = parser.add_argument_group("생성 옵션")
    gen_group.add_argument(
        "--tone",
        help=f"문구 톤. 자유 입력 가능 (참고: {', '.join(TONE_OPTIONS)})",
    )
    gen_group.add_argument(
        "--copy-length",
        choices=("짧게", "보통", "긴"),
        default="보통",
        help="해시태그 개수/글자 수 프리셋. --hashtag-count/--max-chars를 주면 이 값은 무시됩니다.",
    )
    gen_group.add_argument("--hashtag-count", type=int, help="해시태그 개수 직접 지정 (선택).")
    gen_group.add_argument("--max-chars", type=int, help="해시태그 하나당 최대 글자 수 직접 지정 (선택).")

    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.example is not None:
        EXAMPLES[args.example]()
        return

    if not _has_custom_input(args):
        # 아무 입력도 주지 않으면 내장 예시 5개를 전부 보여줍니다.
        for number in sorted(EXAMPLES):
            EXAMPLES[number]()
        return

    if args.info:
        product_info = json.loads(Path(args.info).read_text(encoding="utf-8"))
        # CLI로 개별 필드를 추가로 덮어쓸 수 있게 병합
        override = {
            k: v for k, v in _build_info_from_args(args).items() if v
        }
        product_info.update(override)
    elif args.interactive:
        product_info = _build_info_interactively()
    else:
        product_info = _build_info_from_args(args)

    _run(
        "사용자 입력 기반 추천 해시태그",
        product_info,
        tone=args.tone,
        copy_length=args.copy_length,
        hashtag_count=args.hashtag_count,
        max_chars_per_tag=args.max_chars,
    )


if __name__ == "__main__":
    main()
