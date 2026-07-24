"""
GPU 환경(Colab T4/A100, 로컬 CUDA GPU 등)에서 run_pipeline()을 실행하는
5가지 예시 코드입니다. requirements.txt 설치 + CUDA GPU가 필요합니다.

    pip install -r requirements.txt
    cp .env.example .env   # OPENAI_API_KEY 채워넣기
    python examples/gpu_run_examples.py --example 1

--example 로 1~5 중 하나를 선택해 실행하거나, 인자 없이 실행하면 5개를
순서대로 모두 돌립니다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from adcg.pipeline import run_pipeline  # noqa: E402

SAMPLE_IMAGE = REPO_ROOT / "adcg" / "test_data" / "washing_machine_stand_01.png"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "gpu_examples"


def _check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 찾을 수 없습니다. Colab이라면 "
            "런타임 → 런타임 유형 변경 → GPU 로 설정하세요."
        )
    print(f"[GPU] {torch.cuda.get_device_name(0)} 사용")


def _check_api_key() -> None:
    import os

    if os.environ.get("OPENAI_API_KEY"):
        return

    env_path = REPO_ROOT / ".env"
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다.\n"
        f"  1) {env_path} 파일을 만들고 다음 줄을 추가하세요:\n"
        "     OPENAI_API_KEY=sk-...\n"
        "     (.env.example을 복사해서 써도 됩니다: "
        f"cp {REPO_ROOT / '.env.example'} {env_path})\n"
        "  2) 또는 실행 전에 셸에서 직접 export 하세요:\n"
        "     export OPENAI_API_KEY=sk-...\n"
        f"  현재 .env 탐색 경로: {env_path} "
        f"({'존재함' if env_path.exists() else '존재하지 않음'})"
    )


def _write_info(info: dict) -> Path:
    path = OUTPUT_ROOT / f"{info['store_name']}_{info['product_name']}_info.json".replace(" ", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# 예시 1. 상품을 강하게 강조하는 신제품 광고 (레이아웃 새로 구성)
# ------------------------------------------------------------------
def example_1_new_product_launch() -> None:
    print("\n=== 예시 1: 신제품 소개 (레이아웃 새로 구성, 상품 강조 최대) ===")
    info_path = _write_info({
        "product_name": "수제 버터 크루아상",
        "store_name": "아침빵집",
        "store_type": "동네 베이커리",
        "product_category": "베이커리",
        "product_description": "매일 아침 직접 반죽해 굽는 프랑스산 버터 크루아상",
        "features": ["프랑스산 버터 100% 사용", "매일 아침 소량 생산"],
        "target_customer": "출근길 직장인",
        "promo_target": "",
        "promotion": "오픈 기념 아메리카노 500원 할인",
        "price": "3,800원",
        "tone": "친근한",
        "draft_copy": "",
    })

    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        info_path=info_path,
        output_dir=OUTPUT_ROOT / "01_new_product",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="보통",
        focus_strength=1.0,     # Focus Slider: 상품 강조 최대
        layout_mode="layout",   # 레이아웃 새롭게 구성
        seed=42,
    )
    print("최종 이미지:", result.final_image)


# ------------------------------------------------------------------
# 예시 2. 원본 사진을 그대로 살리는 인테리어/가구 광고
# ------------------------------------------------------------------
def example_2_preserve_original_layout() -> None:
    print("\n=== 예시 2: 가구/인테리어 (원본 레이아웃 유지, 상품·배경 균형) ===")
    info_path = _write_info({
        "product_name": "세탁기 받침대",
        "store_name": "홈리빙마켓",
        "store_type": "소상공인 매장",
        "product_category": "생활/인테리어",
        "product_description": "세탁기 아래에 놓아 층간소음과 진동을 줄여주는 받침대",
        "features": ["층간소음 저감", "미끄럼 방지", "간편 설치"],
        "target_customer": "아파트/빌라 거주 1인 가구",
        "promo_target": "신혼부부, 자취 시작하는 1인 가구",
        "promotion": "",
        "price": "",
        "tone": "미니멀한",
        "draft_copy": "세탁기 밑에 놓기만 해도 층간소음이 줄어요",
    })

    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        info_path=info_path,
        output_dir=OUTPUT_ROOT / "02_preserve_layout",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="짧게",
        focus_strength=0.5,      # 상품과 배경을 균형 있게
        layout_mode="preserve",  # 원본 사진 구도 유지
        seed=7,
    )
    print("최종 이미지:", result.final_image)


# ------------------------------------------------------------------
# 예시 3. 배경/브랜드 무드를 강조하는 고급 브랜드 광고
# ------------------------------------------------------------------
def example_3_brand_mood_focus() -> None:
    print("\n=== 예시 3: 고급 브랜드 (배경/브랜드 무드 강조, 긴 카피) ===")
    info_path = _write_info({
        "product_name": "핸드메이드 가죽 카드지갑",
        "store_name": "아뜰리에 소",
        "store_type": "가죽공방",
        "product_category": "패션잡화",
        "product_description": "이탈리아산 식물성 무두질 가죽으로 한 땀 한 땀 제작한 카드지갑",
        "features": ["이탈리아산 베지터블 가죽", "각인 서비스 제공"],
        "target_customer": "선물을 찾는 30-40대",
        "promo_target": "",
        "promotion": "",
        "price": "89,000원",
        "tone": "고급스러운",
        "draft_copy": "",
    })

    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        info_path=info_path,
        output_dir=OUTPUT_ROOT / "03_brand_mood",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="긴",
        focus_strength=0.3,     # 상품보다 배경/브랜드 무드를 강조
        layout_mode="layout",
        seed=101,
    )
    print("최종 이미지:", result.final_image)


# ------------------------------------------------------------------
# 예시 4. 저사양 GPU(T4 등, VRAM 제한)에서 cpu_offload로 돌리기
# ------------------------------------------------------------------
def example_4_low_vram_cpu_offload() -> None:
    print("\n=== 예시 4: 저사양 GPU (cpu_offload=True, VRAM 절약) ===")
    info_path = _write_info({
        "product_name": "미니 무선 선풍기",
        "store_name": "쿨브리즈",
        "store_type": "온라인 소품샵",
        "product_category": "생활가전",
        "product_description": "휴대가 간편한 저소음 무선 미니 선풍기",
        "features": ["저소음 모터", "USB-C 고속 충전", "3단 풍속 조절"],
        "target_customer": "여름 캠핑족, 사무실 직장인",
        "promo_target": "",
        "promotion": "",
        "price": "29,900원",
        "tone": "위트있는",
        "draft_copy": "",
    })

    # T4처럼 VRAM이 넉넉하지 않은 GPU에서는 cpu_offload=True로
    # 모델 일부를 CPU로 내려서 메모리 부족(OOM)을 방지합니다.
    # (속도는 다소 느려지지만 더 낮은 VRAM에서도 실행 가능)
    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        info_path=info_path,
        output_dir=OUTPUT_ROOT / "04_low_vram",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="보통",
        focus_strength=0.8,
        layout_mode="layout",
        seed=2024,
        cpu_offload=True,
    )
    print("최종 이미지:", result.final_image)


# ------------------------------------------------------------------
# 예시 5. 생성 후 정량 평가까지 함께 실행 (CLIP score / 미학 점수)
# ------------------------------------------------------------------
def example_5_with_evaluation() -> None:
    print("\n=== 예시 5: 생성 + 정량 평가 (clip_score, aesthetic_score) ===")
    info_path = _write_info({
        "product_name": "전동 지게차 부품",
        "store_name": "산업부품마트",
        "store_type": "산업용품 도매",
        "product_category": "산업/공구",
        "product_description": "내구성이 뛰어난 산업용 전동 지게차 교체 부품",
        "features": ["고내구성 소재", "표준 규격 호환"],
        "target_customer": "물류창고, 제조 공장, 건설 현장 관리자",
        "promo_target": "",
        "promotion": "",
        "price": "",
        "tone": "전문적인",
        "draft_copy": "",
    })

    result = run_pipeline(
        image_path=SAMPLE_IMAGE,
        info_path=info_path,
        output_dir=OUTPUT_ROOT / "05_with_evaluation",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="보통",
        focus_strength=0.9,
        layout_mode="layout",
        seed=55,
        evaluate=True,
        eval_metrics=["clip_score", "aesthetic_score"],
    )
    print("최종 이미지:", result.final_image)
    print("평가 결과 JSON:", result.eval_json)


EXAMPLES = {
    1: example_1_new_product_launch,
    2: example_2_preserve_original_layout,
    3: example_3_brand_mood_focus,
    4: example_4_low_vram_cpu_offload,
    5: example_5_with_evaluation,
}


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    _check_api_key()
    _check_gpu()

    parser = argparse.ArgumentParser(description="adcg GPU 실행 예시")
    parser.add_argument(
        "--example",
        type=int,
        choices=sorted(EXAMPLES),
        default=None,
        help="1~5 중 하나만 실행. 생략하면 5개를 모두 순서대로 실행.",
    )
    args = parser.parse_args()

    targets = [args.example] if args.example else sorted(EXAMPLES)
    for number in targets:
        EXAMPLES[number]()


if __name__ == "__main__":
    main()
