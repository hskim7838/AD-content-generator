# -*- coding: utf-8 -*-
"""
step8_compose_banner.py
------------------------------------
step6_evaluate_generated_copy.py가 만든 generated_copy_eval_result.csv(정량 평가 결과)와
generated_copy_results.json(상품별 원본 이미지 경로)을 읽어서,
평가 점수가 가장 좋은 카피를 원본 상품 사진 위에 합성해 배너 이미지(PNG)로 저장한다.

실행 순서:
  1) python copywrite_step4_5.py               (generated_copy_results.json 생성)
  2) python step6_evaluate_generated_copy.py   (generated_copy_eval_result.csv 생성)
  3) python step8_compose_banner.py            (이 파일 — 배너 이미지 합성)

선택 로직 (SELECTION_MODE):
  - "best_per_product" : 상품 하나당 judge_total이 가장 높은 카피 1개만 합성 (기본값)
  - "best_per_tone"    : 상품 x 톤 조합마다 judge_total이 가장 높은 카피를 합성
                         (톤별로 배너 문구/색상 차이를 비교하고 싶을 때)
  - "all"              : eval CSV의 모든 행을 합성 (양이 많을 수 있으니 주의)

폰트:
  한글이 깨지지 않으려면 한글을 지원하는 TTF/TTC 폰트가 필요하다. 아래 FONT_CANDIDATES에서
  자동으로 찾고, 못 찾으면 FONT_PATH 환경변수나 직접 지정한 경로를 쓰도록 안내 메시지를 띄운다.
  (예: macOS는 AppleGothic, Windows는 맑은 고딕, Linux는 Noto Sans CJK KR을 설치해서 사용)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

# ── 0-1. Pillow 설치 확인 ─────────────────────────────────────────────
# 이 파일은 pandas/openai/dotenv 외에 이미지 처리를 위해 Pillow(PIL)가 추가로 필요합니다.
#   pip install pillow
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit(
        "Pillow가 설치되어 있지 않습니다. 아래 명령으로 설치한 뒤 다시 실행하세요.\n"
        "  pip install pillow"
    )

# ══════════════════════════════════════════════════════════════
# 사용자 설정 (실제 브랜드/배너 규격에 맞게 이 블록만 조정하면 됩니다)
# ══════════════════════════════════════════════════════════════

# ── 0-2. 경로/선택 로직 ───────────────────────────────────────────────
EVAL_RESULT_CSV_PATH = "./generated_copy_eval_result.csv"
GENERATED_COPY_RESULTS_PATH = "./generated_copy_results.json"
OUTPUT_DIR = "./generated_banners"

SELECTION_MODE = "best_per_product"  # "best_per_product" | "best_per_tone" | "all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for _p in (EVAL_RESULT_CSV_PATH, GENERATED_COPY_RESULTS_PATH):
    if not os.path.exists(_p):
        raise FileNotFoundError(
            f"{_p} 가 없습니다. copywrite_step4_5.py -> step6_evaluate_generated_copy.py 를 "
            "먼저 순서대로 실행하세요."
        )

# ── 1. 배너 캔버스 크기 ──────────────────────────────────────────────
# 자주 쓰는 SNS 광고 규격 프리셋. CANVAS_PRESET 값만 바꿔서 고르거나,
# CANVAS_SIZE를 (가로, 세로) 픽셀로 직접 지정해도 됩니다.
CANVAS_PRESETS = {
    "square": (1080, 1080),     # 인스타 피드 정사각형
    "portrait": (1080, 1350),   # 인스타 피드 세로형(4:5)
    "story": (1080, 1920),      # 인스타/카카오 스토리, 릴스(9:16)
    "wide": (1200, 628),        # 페이스북/카카오 배너 광고(가로형)
}
CANVAS_PRESET = "square"
CANVAS_SIZE = CANVAS_PRESETS[CANVAS_PRESET]

BOTTOM_BAND_RATIO = 0.40  # 캔버스 아래쪽 몇 %를 카피(텍스트) 영역으로 쓸지 (0.0~1.0)

# ── 2. 톤별 스타일(배색) ─────────────────────────────────────────────
# bg    : 카피 영역 배경색 (R, G, B, Alpha) — Alpha가 낮을수록 사진이 비쳐 보임
# text  : 타이틀/서브카피 글자색
# accent: 가격 강조색 + CTA 버튼 색
# 매장 로고/브랜드 컬러에 맞춰 값만 바꾸면 즉시 반영됩니다.
TONE_STYLES = {
    "미니멀":     {"bg": (255, 255, 255, 235), "text": (20, 20, 20),   "accent": (20, 20, 20)},
    "강조형":     {"bg": (20, 20, 20, 235),     "text": (255, 255, 255), "accent": (230, 57, 70)},
    "친근한":     {"bg": (255, 233, 210, 235),  "text": (90, 55, 30),   "accent": (255, 140, 66)},
    "고급스러운": {"bg": (18, 24, 38, 235),      "text": (240, 225, 180), "accent": (198, 161, 91)},
    "위트있는":   {"bg": (255, 214, 10, 235),    "text": (20, 20, 20),   "accent": (20, 20, 20)},
}
DEFAULT_STYLE = {"bg": (20, 20, 20, 220), "text": (255, 255, 255), "accent": (255, 140, 66)}

# ── 3. 한글 폰트 자동 탐색 ───────────────────────────────────────────
# 우선순위: FONT_PATH 환경변수 > ./fonts/ 폴더(프로젝트에 직접 넣어둔 폰트) >
#           OS 기본 한글 폰트(macOS: AppleGothic 등, Windows: 맑은 고딕, Linux: Noto Sans CJK)
#
# 한글이 깨지거나(네모/물음표) 폰트를 못 찾는다는 에러가 나면:
#   1) 프로젝트 폴더에 ./fonts/ 를 만들고 한글 TTF/OTF 파일(Noto Sans KR, 나눔고딕 등)을 넣으세요.
#   2) 또는 실행 전에 FONT_PATH 환경변수로 폰트 경로를 직접 지정하세요.
#      예) FONT_PATH=/path/to/font.ttf python step8_compose_banner.py
FONTS_DIR = "./fonts"
os.makedirs(FONTS_DIR, exist_ok=True)

FONT_CANDIDATES = [
    os.environ.get("FONT_PATH", ""),
    # 프로젝트 폴더에 직접 넣어둔 폰트 (파일명은 자유롭게 넣어도 되지만, 예시로 흔한 이름들을 먼저 찾음)
    os.path.join(FONTS_DIR, "NanumSquareRoundEB.ttf"),
    os.path.join(FONTS_DIR, "NotoSansKR-Bold.otf"),
    os.path.join(FONTS_DIR, "NotoSansKR-Bold.ttf"),
    # macOS 기본 한글 폰트
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    # Windows 기본 한글 폰트(맑은 고딕)
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    # Linux (Noto Sans CJK — 배포판에 따라 경로가 다를 수 있음)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]
# ./fonts/ 폴더 안에 위 후보 이름과 다른 파일을 넣었어도 자동으로 잡히도록,
# 폴더 안의 .ttf/.otf/.ttc 파일도 후보에 추가
if os.path.isdir(FONTS_DIR):
    FONT_CANDIDATES += [
        os.path.join(FONTS_DIR, f)
        for f in sorted(os.listdir(FONTS_DIR))
        if f.lower().endswith((".ttf", ".otf", ".ttc"))
    ]

FONT_PATH = next((p for p in FONT_CANDIDATES if p and os.path.exists(p)), None)

if FONT_PATH is None:
    raise FileNotFoundError(
        "한글을 지원하는 폰트를 찾지 못했습니다.\n"
        f"'{FONTS_DIR}' 폴더를 만들어뒀으니, 아래 중 하나로 해결하세요:\n"
        f"  1) Noto Sans KR, 나눔고딕 등 한글 TTF/OTF 폰트 파일을 '{FONTS_DIR}/' 안에 넣기\n"
        "  2) 실행 전에 FONT_PATH 환경변수로 폰트 경로 직접 지정하기\n"
        "     예) FONT_PATH=/path/to/font.ttf python step8_compose_banner.py\n"
        "  (macOS라면 보통 AppleGothic이 자동으로 잡히는데, 안 잡혔다면 시스템에 없는 것이니 "
        "위 1)번으로 폰트를 직접 넣어주세요.)"
    )


def get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


# ── 3. 텍스트 유틸 ───────────────────────────────────────────────────
def fit_font_size(
    draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, min_size: int = 18
) -> ImageFont.FreeTypeFont:
    """max_width 안에 한 줄로 들어가도록 폰트 크기를 줄여가며 맞춘다."""
    size = start_size
    while size > min_size:
        font = get_font(size)
        w = draw.textlength(text, font=font)
        if w <= max_width:
            return font
        size -= 2
    return get_font(min_size)


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int = 2
) -> list[str]:
    """공백 기준으로 줄바꿈하되, 공백이 거의 없는 한글 문장도 글자 단위로 보정."""
    if not text:
        return []
    words = text.split(" ")
    if len(words) < 2:
        # 공백이 없으면 글자 단위로 줄바꿈
        words = list(text)
        sep = ""
    else:
        sep = " "

    lines, current = [], ""
    for w in words:
        candidate = (current + sep + w) if current else w
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines[:max_lines]


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


# ── 4. 배너 한 장 합성 ────────────────────────────────────────────────
def compose_banner(image_path: str, copy_row: dict, out_path: str) -> None:
    tone = copy_row.get("tone", "")
    style = TONE_STYLES.get(tone, DEFAULT_STYLE)

    # 4-1. 원본 이미지를 정사각 캔버스에 꽉 차게(cover) 배치
    base = Image.open(image_path).convert("RGB")
    base_ratio = base.width / base.height
    canvas_ratio = CANVAS_SIZE[0] / CANVAS_SIZE[1]
    if base_ratio > canvas_ratio:
        new_h = CANVAS_SIZE[1]
        new_w = int(new_h * base_ratio)
    else:
        new_w = CANVAS_SIZE[0]
        new_h = int(new_w / base_ratio)
    base = base.resize((new_w, new_h))
    left = (new_w - CANVAS_SIZE[0]) // 2
    top = (new_h - CANVAS_SIZE[1]) // 2
    base = base.crop((left, top, left + CANVAS_SIZE[0], top + CANVAS_SIZE[1]))

    canvas = base.convert("RGBA")

    # 4-2. 하단 카피 영역(반투명 색 배경 박스)
    band_h = int(CANVAS_SIZE[1] * BOTTOM_BAND_RATIO)
    band_top = CANVAS_SIZE[1] - band_h
    overlay = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, band_top, CANVAS_SIZE[0], CANVAS_SIZE[1]], fill=style["bg"])
    canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)
    pad_x = 56
    max_text_width = CANVAS_SIZE[0] - pad_x * 2
    y = band_top + 36

    # 4-3. 타이틀
    title = str(copy_row.get("title") or "").strip()
    if title:
        title_font = fit_font_size(draw, title, max_text_width, start_size=64, min_size=32)
        for line in wrap_text_to_width(draw, title, title_font, max_text_width, max_lines=2):
            draw.text((pad_x, y), line, font=title_font, fill=style["text"])
            y += title_font.size + 10
        y += 8

    # 4-4. 서브카피
    subtitle = str(copy_row.get("subtitle") or "").strip()
    if subtitle:
        subtitle_font = fit_font_size(draw, subtitle, max_text_width, start_size=34, min_size=20)
        for line in wrap_text_to_width(draw, subtitle, subtitle_font, max_text_width, max_lines=2):
            draw.text((pad_x, y), line, font=subtitle_font, fill=style["text"])
            y += subtitle_font.size + 8
        y += 6

    # 4-5. 가격 강조 문구
    price = str(copy_row.get("price") or "").strip()
    if price and price.lower() != "nan":
        price_font = fit_font_size(draw, price, max_text_width, start_size=30, min_size=20)
        draw.text((pad_x, y), price, font=price_font, fill=style["accent"])
        y += price_font.size + 14

    # 4-6. CTA 버튼
    cta = str(copy_row.get("cta") or "").strip()
    if cta:
        cta_font = get_font(28)
        text_w = draw.textlength(cta, font=cta_font)
        btn_w = int(text_w) + 64
        btn_h = 64
        btn_x0 = pad_x
        btn_y0 = CANVAS_SIZE[1] - 36 - btn_h
        btn_y0 = max(btn_y0, y + 10)
        rounded_rect(
            draw,
            [btn_x0, btn_y0, btn_x0 + btn_w, btn_y0 + btn_h],
            radius=btn_h // 2,
            fill=style["accent"],
        )
        # 버튼 배경색과 대비되는 글자색 선택 (밝기 기준)
        r, g, b = style["accent"][:3]
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        btn_text_color = (20, 20, 20) if brightness > 150 else (255, 255, 255)
        text_x = btn_x0 + (btn_w - text_w) / 2
        text_y = btn_y0 + (btn_h - cta_font.size) / 2 - 4
        draw.text((text_x, text_y), cta, font=cta_font, fill=btn_text_color)

    canvas.convert("RGB").save(out_path, quality=95)


# ── 5. 합성할 카피 선택 + 실행 ────────────────────────────────────────
eval_df = pd.read_csv(EVAL_RESULT_CSV_PATH)
if eval_df.empty:
    raise RuntimeError(f"{EVAL_RESULT_CSV_PATH} 에 평가 결과가 없습니다.")

with open(GENERATED_COPY_RESULTS_PATH, encoding="utf-8") as f:
    products = json.load(f)["products"]

if SELECTION_MODE == "best_per_product":
    selected_df = eval_df.loc[eval_df.groupby("product_id")["judge_total"].idxmax()]
elif SELECTION_MODE == "best_per_tone":
    selected_df = eval_df.loc[eval_df.groupby(["product_id", "tone"])["judge_total"].idxmax()]
elif SELECTION_MODE == "all":
    selected_df = eval_df
else:
    raise ValueError(f"알 수 없는 SELECTION_MODE: {SELECTION_MODE}")

print(f"\n=== [STEP 8] 합성 대상 {len(selected_df)}건 (SELECTION_MODE={SELECTION_MODE}) ===")
print(f"사용 폰트: {FONT_PATH}")

saved_paths = []
for i, (_, row) in enumerate(selected_df.iterrows(), 1):
    copy_row = row.to_dict()
    product_id = copy_row.get("product_id")
    product_info = products.get(product_id)
    if product_info is None or not product_info.get("image"):
        print(f"[{i}] ⚠ product_id={product_id} 의 원본 이미지 경로를 찾을 수 없어 건너뜁니다.")
        continue

    image_path = product_info["image"]
    if not os.path.exists(image_path):
        print(f"[{i}] ⚠ 이미지 파일이 없습니다: {image_path} -> 건너뜁니다.")
        continue

    model = str(copy_row.get("model", "model")).replace(" ", "_")
    tone = str(copy_row.get("tone", "tone"))
    out_name = f"{product_id}_{tone}_{model}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        compose_banner(image_path, copy_row, out_path)
    except Exception as e:
        print(f"[{i}] ❌ 합성 실패 (product_id={product_id}, tone={tone}, model={model}): {e}")
        continue

    print(f"[{i}] ✅ {out_path}  (judge_total={copy_row.get('judge_total')})")
    saved_paths.append(out_path)

print(f"\n=== [STEP 8] 완료: {len(saved_paths)}장 저장 -> {OUTPUT_DIR}/ ===")
