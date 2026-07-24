from __future__ import annotations

import json
import re
from time import perf_counter

from openai import OpenAI

from ..config import TONE_OPTIONS

HASHTAG_INPUT_FIELDS = (
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

# "글 길이" 선택지와 동일한 체계를 사용해 해시태그 개수/길이를 결정합니다.
# UI의 "보통" / "긴" 옵션과 맞춰 놓았고, 필요하면 값을 자유롭게 조정할 수 있습니다.
HASHTAG_LENGTH_PRESETS = {
    "짧게": {"count": 4, "max_chars": 8},
    "보통": {"count": 6, "max_chars": 10},
    "긴": {"count": 10, "max_chars": 14},
}

DEFAULT_PRESET_KEY = "보통"

HASHTAG_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]")


def _resolve_preset(copy_length: str) -> dict:
    return HASHTAG_LENGTH_PRESETS.get(
        copy_length,
        HASHTAG_LENGTH_PRESETS[DEFAULT_PRESET_KEY],
    )


def _normalize_tag(raw_tag: str, max_chars: int) -> str | None:
    body = HASHTAG_PATTERN.sub("", str(raw_tag or ""))
    body = body.strip()

    if not body:
        return None

    body = body[:max_chars]

    return f"#{body}"


def _dedupe_tags(tags: list[str], limit: int) -> list[str]:
    seen = set()
    deduped = []

    for tag in tags:
        key = tag.lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(tag)

        if len(deduped) >= limit:
            break

    return deduped


def generate_ad_hashtags(
    product_info: dict,
    background_prompt: str = "",
    tone: str | None = None,
    copy_length: str = DEFAULT_PRESET_KEY,
    hashtag_count: int | None = None,
    max_chars_per_tag: int | None = None,
    model: str = "gpt-5.4-nano",
    client: OpenAI | None = None,
) -> dict:
    """Recommend Korean advertising hashtags grounded in store/product facts.

    Args:
        product_info: 상품/매장 정보 딕셔너리 (product_info.json과 동일한 스키마).
            product_info에 없는 필드는 빈 값으로 취급되므로, 어떤 조합의 정보를
            넣어도(일부만 채워도) 안전하게 동작합니다.
        background_prompt: 생성된 배경 장면 설명(선택). 문맥에 맞는 해시태그 추천에 사용.
        tone: 문구 톤. 자유 문자열 아무 값이나 가능. 생략 시 product_info["tone"] 값을,
            그마저 없으면 TONE_OPTIONS[0]을 사용.
        copy_length: "짧게" / "보통" / "긴" 중 하나. 해시태그 개수와 글자 수 제한 프리셋을
            결정. 목록에 없는 값이 들어와도 에러 없이 "보통"으로 대체됨.
        hashtag_count: 생성할 해시태그 개수를 직접 지정(선택). 지정하면 copy_length
            프리셋의 개수 대신 이 값을 사용.
        max_chars_per_tag: 해시태그 하나당 최대 글자 수를 직접 지정(선택). 지정하면
            copy_length 프리셋의 글자 수 제한 대신 이 값을 사용.
        model: 사용할 모델명.
        client: 재사용할 OpenAI 클라이언트(선택). 생략 시 새로 생성.
    """
    preset = _resolve_preset(copy_length)
    hashtag_count = hashtag_count if hashtag_count else preset["count"]
    max_chars = max_chars_per_tag if max_chars_per_tag else preset["max_chars"]

    resolved_tone = tone or product_info.get("tone") or TONE_OPTIONS[0]

    request_data = {
        field: product_info.get(field)
        for field in HASHTAG_INPUT_FIELDS
    }
    request_data["tone"] = resolved_tone
    request_data["background_prompt"] = background_prompt

    schema = {
        "type": "object",
        "properties": {
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": hashtag_count,
                "maxItems": hashtag_count,
            },
        },
        "required": ["hashtags"],
        "additionalProperties": False,
    }

    prompt = (
        "Recommend Korean advertising hashtags using only the supplied "
        "store/product facts. Do not invent prices, discounts, benefits, "
        "certifications, or locations that are not present in the input. "
        f"Return exactly {hashtag_count} hashtags. Each hashtag body "
        f"(without the '#') must be {max_chars} Korean characters or "
        "fewer, contain no spaces or punctuation, and must not duplicate "
        "another hashtag. Mix short brand/store-name tags, product/category "
        "tags, tone-driven mood tags, and audience tags. If promo_target is "
        "present, prioritize it over target_customer when framing audience "
        "tags, since it names who this specific campaign is aimed at. If "
        "draft_copy is present, you may draw keywords or phrasing cues from "
        "it, but do not quote it verbatim as a hashtag. Return the hashtag "
        "text only, without the leading '#' character.\n\n"
        + json.dumps(request_data, ensure_ascii=False, indent=2)
    )

    started_at = perf_counter()
    client = client or OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a Korean social-media advertising copywriter who "
            "writes concise, on-brand hashtags using only supplied facts."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "advertising_hashtags",
                "strict": True,
                "schema": schema,
            }
        },
    )

    parsed = json.loads(response.output_text)
    raw_tags = parsed.get("hashtags", [])

    normalized = [
        tag
        for tag in (
            _normalize_tag(raw_tag, max_chars) for raw_tag in raw_tags
        )
        if tag
    ]
    hashtags = _dedupe_tags(normalized, hashtag_count)

    return {
        "hashtags": hashtags,
        "tone": resolved_tone,
        "copy_length": copy_length,
        "hashtag_count": len(hashtags),
        "max_chars_per_tag": max_chars,
        "model": model,
        "latency_sec": round(perf_counter() - started_at, 2),
    }
