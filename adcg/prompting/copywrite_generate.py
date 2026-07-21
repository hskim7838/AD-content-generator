from __future__ import annotations

import json
from time import perf_counter

from openai import OpenAI

COPY_INPUT_FIELDS = (
    "product_name",
    "store_name",
    "store_type",
    "product_category",
    "product_description",
    "price",
    "tone",
    "seller_description",
    "store_info",
    "reviews",
    "focus",
    "additional_request",
)


def generate_ad_copies(
    product_info: dict,
    background_prompt: str,
    model: str,
    count: int = 1,
    client=None,
) -> list[dict]:
    """Generate multiple Korean ad-copy variants in one API request."""
    count = int(count)
    if count < 1:
        raise ValueError("count는 1 이상이어야 합니다.")

    request_data = {
        field: product_info.get(field)
        for field in COPY_INPUT_FIELDS
    }
    request_data["background_prompt"] = background_prompt
    copy_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "price": {"type": "string"},
            "cta": {"type": "string"},
        },
        "required": ["title", "subtitle", "price", "cta"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "copies": {
                "type": "array",
                "items": copy_schema,
                "minItems": count,
                "maxItems": count,
            }
        },
        "required": ["copies"],
        "additionalProperties": False,
    }
    prompt = (
        f"아래 정보를 바탕으로 서로 다른 한국어 광고 문구를 정확히 {count}개 작성하세요. "
        "각 문구는 자연스럽고 간결해야 합니다. "
        "확인할 수 없는 가격, 할인율, 효능은 만들지 마세요. "
        "가격 정보가 없으면 price는 빈 문자열로 반환하세요.\n\n"
        + json.dumps(request_data, ensure_ascii=False, indent=2)
    )

    started_at = perf_counter()
    client = client or OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "당신은 상품 광고 카피라이터입니다. 서로 중복되지 않는 문구를 "
            "만들고 title, subtitle, price, cta를 각각 짧고 명확하게 작성하세요."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "advertising_copy",
                "strict": True,
                "schema": schema,
            }
        },
    )
    result = json.loads(response.output_text)
    copies = result.get("copies", [])

    if len(copies) != count:
        raise ValueError(
            f"광고 카피를 {count}개 요청했지만 {len(copies)}개를 받았습니다."
        )

    latency_sec = round(perf_counter() - started_at, 2)
    for copy in copies:
        copy["model"] = model
        copy["latency_sec"] = latency_sec

    return copies


def generate_ad_copy(
    product_info: dict,
    background_prompt: str,
    model: str,
    client=None,
) -> dict:
    """Backward-compatible single-copy API."""
    return generate_ad_copies(
        product_info=product_info,
        background_prompt=background_prompt,
        model=model,
        count=1,
        client=client,
    )[0]
