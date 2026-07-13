# -*- coding: utf-8 -*-
"""
banner_utils.py
-----------------
compare_copy_llms.py 의 결과(title/subtitle/price/cta)를 제품 사진 위에 합성해
광고 배너 이미지를 만드는 공통 함수 모음.

두 가지 합성 함수 제공:
  - compose_banner_simple(...) : 카피만 그대로 얹는 기본 버전
  - compose_banner(...)        : focusing_degree/background_intensity/preservation_degree
                                   수치 옵션을 블러·색보정·오버레이로 시각적으로 반영하는 버전
                                   (실제 ControlNet/SD를 붙이기 전 임시 프록시 효과)

Colab 버전과 다른 점: 폰트를 못 찾았을 때 `apt-get install fonts-nanum`으로 자동 설치를
시도하던 부분을, OS별(Windows/macOS/Linux) 후보 경로를 먼저 탐색하도록 확장했습니다.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ── 0. 한글 폰트 자동 탐색 ─────────────────────────────────────────
_TTC_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
     "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1),  # Linux, index=1 -> KR
]
_TTF_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux (나눔고딕 설치 시)
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "C:/Windows/Fonts/malgun.ttf",                             # Windows (맑은 고딕)
    "C:/Windows/Fonts/malgunbd.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",     # macOS
]


def _resolve_korean_font() -> dict | None:
    """환경에 따라 한글 지원 폰트 경로/인덱스를 찾아 반환. 못 찾으면 None."""
    for reg, bold, idx in _TTC_CANDIDATES:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                ImageFont.truetype(reg, 20, index=idx)
                return {"type": "ttc", "regular": reg, "bold": bold, "index": idx}
            except Exception:
                continue

    reg_path = next((p for p in _TTF_CANDIDATES if os.path.exists(p)), None)
    if reg_path:
        bold_path = next(
            (p for p in _TTF_CANDIDATES if "Bold" in p or "bd" in p.lower()
             if os.path.exists(p)),
            reg_path,
        )
        return {"type": "ttf", "regular": reg_path, "bold": bold_path, "index": 0}

    # 리눅스 계열이고 apt가 있으면 설치 시도 (Windows/macOS 로컬 환경에서는 무시됨)
    if os.name == "posix" and shutil_which("apt-get"):
        print("한글 폰트를 찾지 못해 설치를 시도합니다 (fonts-nanum)...")
        os.system("sudo apt-get -qq update > /dev/null 2>&1")
        os.system("sudo apt-get -qq install -y fonts-nanum > /dev/null 2>&1")
        nanum_reg = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        nanum_bold = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
        if os.path.exists(nanum_reg):
            bold_path = nanum_bold if os.path.exists(nanum_bold) else nanum_reg
            return {"type": "ttf", "regular": nanum_reg, "bold": bold_path, "index": 0}

    print(
        "⚠ 한글 폰트를 찾지 못했습니다. 그래프/배너의 한글이 깨져 보일 수 있습니다.\n"
        "   나눔고딕(Linux) / 맑은 고딕(Windows, 기본 내장) / Apple Gothic(macOS) 등을 "
        "설치한 뒤 banner_utils.py의 _TTF_CANDIDATES에 경로를 추가하세요."
    )
    return None


def shutil_which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


_FONT_INFO = _resolve_korean_font()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if _FONT_INFO is None:
        return ImageFont.load_default()
    path = _FONT_INFO["bold"] if bold else _FONT_INFO["regular"]
    try:
        if _FONT_INFO["type"] == "ttc":
            return ImageFont.truetype(path, size, index=_FONT_INFO["index"])
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ── 1. 텍스트 줄바꿈 ────────────────────────────────────────────────
def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """긴 텍스트를 그려질 폭(max_width) 기준으로 자동 줄바꿈."""
    lines, current = [], ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=fnt)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = test
    if current:
        lines.append(current)
    return lines


# ── 2. 그라데이션 오버레이 ───────────────────────────────────────────
def make_gradient_overlay_simple(size: tuple[int, int], top_alpha: int = 0, bottom_alpha: int = 200) -> Image.Image:
    """하단이 어두워지는 세로 그라데이션 오버레이 (텍스트 가독성용, 기본 버전)."""
    w, h = size
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    gradient = Image.new("L", (1, h), color=0)
    start_y = int(h * 0.55)
    for y in range(h):
        if y < start_y:
            alpha = top_alpha
        else:
            ratio = (y - start_y) / (h - start_y)
            alpha = int(top_alpha + (bottom_alpha - top_alpha) * ratio)
        gradient.putpixel((0, y), alpha)
    alpha_mask = gradient.resize((w, h))
    black = Image.new("RGBA", size, (0, 0, 0, 255))
    return Image.composite(black, overlay, alpha_mask)


def make_gradient_overlay(size: tuple[int, int], background_intensity: float) -> Image.Image:
    """background_intensity(0~1)에 비례해 하단 어두운 정도가 커지는 그라데이션 오버레이."""
    w, h = size
    bottom_alpha = int(100 + background_intensity * 130)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    gradient = Image.new("L", (1, h), 0)
    start_y = int(h * 0.55)
    for y in range(h):
        alpha = 0 if y < start_y else int(bottom_alpha * (y - start_y) / (h - start_y))
        gradient.putpixel((0, y), alpha)
    alpha_mask = gradient.resize((w, h))
    black = Image.new("RGBA", size, (0, 0, 0, 255))
    return Image.composite(black, overlay, alpha_mask)


# ── 3. 수치 옵션 -> 시각적 프록시 효과 (진짜 ControlNet 붙기 전 임시) ──
def apply_numeric_effects(
    base_img: Image.Image,
    focusing_degree: float,
    background_intensity: float,
    preservation_degree: float,
) -> Image.Image:
    img = base_img.convert("RGB")
    w, h = img.size

    if focusing_degree > 0.05:
        blur_radius = focusing_degree * 10
        blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = Image.new("L", (w, h), 0)
        mdraw = ImageDraw.Draw(mask)
        cx, cy = w * 0.5, h * 0.42
        max_r = (w ** 2 + h ** 2) ** 0.5 * 0.35
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                val = min(255, int(255 * (dist / max_r) ** 1.5))
                mdraw.rectangle([x, y, x + 4, y + 4], fill=val)
        mask = mask.filter(ImageFilter.GaussianBlur(20))
        img = Image.composite(blurred, img, mask)

    style_shift = 1.0 - preservation_degree
    if style_shift > 0.02:
        img = ImageEnhance.Color(img).enhance(1.0 + style_shift * 0.8)
        img = ImageEnhance.Contrast(img).enhance(1.0 + style_shift * 0.3)
        r, g, b = img.split()
        tint_strength = int(style_shift * 25)
        r = r.point(lambda v: min(255, v + tint_strength))
        b = b.point(lambda v: max(0, v - tint_strength // 2))
        img = Image.merge("RGB", (r, g, b))

    return img


# ── 4. 배너 합성 (기본 버전 — copy 텍스트만 그대로 얹기) ───────────────
def compose_banner_simple(product_img_path: str, copy: dict, out_path: str) -> str:
    base = Image.open(product_img_path).convert("RGBA")
    w, h = base.size

    overlay = make_gradient_overlay_simple((w, h))
    base = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(base)
    margin = int(w * 0.08)
    max_text_width = w - margin * 2

    price_font = font(int(w * 0.055), bold=True)
    price_text = copy.get("price", "") or "가격 문의"
    bbox = draw.textbbox((0, 0), price_text, font=price_font)
    badge_w, badge_h = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 24
    badge_x, badge_y = w - badge_w - margin, int(h * 0.05)
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=badge_h // 2, fill=(255, 255, 255, 235),
    )
    draw.text((badge_x + 20, badge_y + 12), price_text, font=price_font, fill=(60, 40, 20, 255))

    title_font = font(int(w * 0.09), bold=True)
    title_lines = wrap_text(draw, copy["title"], title_font, max_text_width)

    subtitle_font = font(int(w * 0.045))
    subtitle_lines = wrap_text(draw, copy["subtitle"], subtitle_font, max_text_width)

    cta_font = font(int(w * 0.04), bold=True)
    cta_text = "→ " + copy["cta"]

    cursor_y = h - margin
    cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
    cursor_y -= (cta_bbox[3] - cta_bbox[1])
    draw.text((margin, cursor_y), cta_text, font=cta_font, fill=(255, 210, 130, 255))
    cursor_y -= 18

    for line in reversed(subtitle_lines):
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        cursor_y -= (bbox[3] - bbox[1] + 6)
        draw.text((margin, cursor_y), line, font=subtitle_font, fill=(240, 235, 225, 255))
    cursor_y -= 14

    for line in reversed(title_lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        cursor_y -= (bbox[3] - bbox[1] + 10)
        draw.text((margin, cursor_y), line, font=title_font, fill=(255, 255, 255, 255))

    base.convert("RGB").save(out_path, quality=95)
    return out_path


# ── 5. 배너 합성 (수치 옵션 반영 버전 — combo 필요) ────────────────────
def compose_banner(product_img_path: str, copy: dict, combo: dict, out_path: str) -> str:
    styled = apply_numeric_effects(
        Image.open(product_img_path),
        combo["focusing_degree"], combo["background_intensity"], combo["preservation_degree"],
    ).convert("RGBA")
    w, h = styled.size

    overlay = make_gradient_overlay((w, h), combo["background_intensity"])
    base = Image.alpha_composite(styled, overlay)
    draw = ImageDraw.Draw(base)
    margin = int(w * 0.08)
    max_text_width = w - margin * 2

    price_font = font(int(w * 0.055), bold=True)
    price_text = copy.get("price") or ""
    bbox = draw.textbbox((0, 0), price_text, font=price_font)
    badge_w, badge_h = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 24
    badge_x, badge_y = w - badge_w - margin, int(h * 0.05)
    draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                            radius=badge_h // 2, fill=(255, 255, 255, 235))
    draw.text((badge_x + 20, badge_y + 12), price_text, font=price_font, fill=(60, 40, 20, 255))

    tag_font = font(int(w * 0.026))
    tag_text = (f"model:{copy['model']}  tone:{combo['tone']}  focus:{combo['focusing_degree']}  "
                f"bg:{combo['background_intensity']}  preserve:{combo['preservation_degree']}")
    draw.rectangle([0, 0, w, int(h * 0.035)], fill=(0, 0, 0, 160))
    draw.text((10, 4), tag_text, font=tag_font, fill=(255, 255, 255, 255))

    title_font = font(int(w * 0.085), bold=True)
    subtitle_font = font(int(w * 0.042))
    cta_font = font(int(w * 0.038), bold=True)

    title_lines = wrap_text(draw, copy["title"], title_font, max_text_width)
    subtitle_lines = wrap_text(draw, copy["subtitle"], subtitle_font, max_text_width)
    cta_text = "→ " + copy["cta"]

    cursor_y = h - margin
    cta_bbox = draw.textbbox((0, 0), cta_text, font=cta_font)
    cursor_y -= (cta_bbox[3] - cta_bbox[1])
    draw.text((margin, cursor_y), cta_text, font=cta_font, fill=(255, 210, 130, 255))
    cursor_y -= 16

    for line in reversed(subtitle_lines):
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        cursor_y -= (bbox[3] - bbox[1] + 6)
        draw.text((margin, cursor_y), line, font=subtitle_font, fill=(240, 235, 225, 255))
    cursor_y -= 12

    for line in reversed(title_lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        cursor_y -= (bbox[3] - bbox[1] + 10)
        draw.text((margin, cursor_y), line, font=title_font, fill=(255, 255, 255, 255))

    base.convert("RGB").save(out_path, quality=95)
    return out_path
