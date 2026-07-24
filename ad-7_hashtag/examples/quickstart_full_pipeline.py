"""
전체 파이프라인(이미지 생성 + 카피 + 해시태그) 실행 예제.

requirements.txt 전체 설치 + CUDA GPU가 필요합니다.
(카피/해시태그만 필요하면 quickstart_copy_and_hashtags.py를 사용하세요.)

실행:
    pip install -r requirements.txt
    cp .env.example .env   # OPENAI_API_KEY 채워넣기
    python examples/quickstart_full_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from adcg.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    result = run_pipeline(
        image_path=REPO_ROOT / "adcg" / "test_data" / "washing_machine_stand_01.png",
        info_path=REPO_ROOT / "examples" / "sample_product_info.json",
        output_dir=REPO_ROOT / "outputs" / "pipeline_demo",
        gpt_model="gpt-5.4-nano",
        copy_count=1,
        copy_length="보통",     # 카피/해시태그 분량 프리셋
        focus_strength=0.8,     # 상품 강조 정도 (Focus Slider)
        layout_mode="layout",   # "layout"=새롭게 구성 / "preserve"=원본 유지
        seed=42,
    )

    print("최종 이미지:", result.final_image)
    print("카피 JSON:", result.copy_json)
    print("해시태그 JSON:", result.hashtag_json)


if __name__ == "__main__":
    main()
