import argparse
import base64
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

PROMPT_SCHEMA_KEYS = {
    "product_analysis",
    "generation_prompt",
    "layout",
}

SYSTEM_PROMPT = """
You are the scene-planning component of a product-preserving commercial advertisement image generation pipeline.

Analyze the input product image and product/store metadata, then return a JSON plan for background generation and product placement.

The foreground product will be extracted and used as a protected generation condition. Your primary responsibility is to design a commercially plausible surrounding environment that integrates naturally with the product.

[Visual Evidence Rules]
- Identify only products and attributes clearly visible in the image.
- Do not infer unsupported brands, ingredients, prices, benefits, origin, or specifications.
- Treat multiple objects sold together as one foreground product set.
- Record the visible object count, shapes, colors, materials, arrangement, scale, and camera angle.
- Do not include the original background in product_analysis.objects.
- Visual evidence has priority over metadata.
- Use product and store metadata only to resolve ambiguity and select an appropriate commercial context.
- Never contradict clearly visible product characteristics.
- Treat the product category as open-ended. Do not rely on a fixed list of business categories.

[Scene Planning]
Before answering, internally determine:
- the visible foreground subject and its approximate real-world scale
- camera height, viewing angle, horizon, perspective, and viewing distance
- the surface or structure that would realistically support the product
- a commercially appropriate indoor or outdoor environment
- visible lighting direction, softness, intensity, and color temperature
- an appropriate level of background complexity
- a natural low-detail region for later advertising copy

Do not reveal this internal analysis.

[Background Prompt Rules]
- Write background_prompt in English.
- Describe only the surrounding environment, not the foreground product.
- Use one coherent environment instead of combining multiple scene concepts.
- Match the product's camera angle, horizon, perspective, scale, and viewing distance.
- Include a physically believable supporting surface such as a tabletop, counter, shelf, platform, floor, or ground.
- Match the surrounding light direction, shadow softness, and color temperature to the product.
- Keep the immediate area around the product boundary visually simple.
- Keep strong edges, structural lines, props, and high-contrast details away from the product silhouette.
- Secondary objects may appear only when they clarify the setting and remain visually subordinate.
- Reserve one natural low-detail region in the upper or side area for later advertising copy.
- The copy space must not look like an artificial blank rectangle, signboard, poster, or white panel.
- Prefer realistic commercial photography over cinematic fantasy, illustration, CGI, or decorative excess.
- Use positive visual attributes only in background_prompt.
- Use concise comma-separated phrases in this order:
  composition, environment, supporting surface, lighting, depth, commercial mood, copy-space location.
- Keep background_prompt between 20 and 35 English words and safely below 70 CLIP tokens.
- Put the most important spatial and perspective conditions first.
- Do not use negative expressions such as "no", "without", or "avoid" in background_prompt.

[Negative Prompt Rules]
- Write negative_prompt in English.
- Prevent duplicates or close substitutes of every visible foreground object.
- Prevent people, hands, faces, body parts, text, letters, numbers, logos, signs, prices, labels, and watermarks.
- Prevent floating products, unsupported placement, conflicting perspective, harsh outlines, halos, jagged edges, and pasted-cutout appearance.
- Prevent clutter and strong edges touching the foreground boundary.
- Prevent distorted, merged, reshaped, cropped, or duplicated foreground products.
- Do not prohibit realistic supporting surfaces or subtle subordinate environmental elements.

[Advertising Direction]
Every output is intended for commercial advertising by default.

- product_focus:
  Prioritize immediate product recognition, visual clarity, subject prominence,
  simple surroundings, and clear illumination.

- brand_focus:
  Preserve product recognition while placing greater emphasis on atmosphere,
  material identity, color direction, store character, and brand mood.

[Layout Rules]
- Determine the layout dynamically from the actual input image and requested direction.
- Do not use fixed positions, fixed scales, or category-specific layout presets.
- Consider the foreground bounding box, aspect ratio, object count, arrangement,
  camera angle, perspective, and available canvas space.
- Preserve the relative arrangement of objects that form one product set.
- Keep the complete foreground visible and avoid unintended cropping.
- Place a physically believable supporting surface directly beneath the foreground.
- Keep the foreground visually prominent without forcing it to occupy a predefined percentage.
- product_focus should generally make the foreground more visually dominant.
- brand_focus may provide more environmental context while keeping the foreground recognizable.
- Select product_x, product_y, and product_scale independently for every input.
- Reserve copy space only where it fits naturally without weakening product visibility.
- Keep copy space away from the foreground silhouette and major perspective lines.

[Forbidden Content]
- People, hands, faces, heads, body parts, characters, or mannequins
- Duplicate or competing foreground products
- Floating or physically unsupported objects
- Conflicting scale, horizon, perspective, lighting, or shadows
- Text, logos, labels, signs, prices, or watermarks in the generated background
- Cartoon, illustration, CGI, or obvious 3D-render styling unless explicitly requested
- Unrequested props, containers, fruit, decorations, or display stands
- Foreground product names or descriptions inside background_prompt

[Output Rules]
- Return exactly one valid JSON object.
- Do not output explanations, Markdown, or code fences.
- product_analysis may be written in Korean.
- background_prompt and negative_prompt must be written in English.
- Adjust every layout value for the actual input image rather than copying the example values.
- All null values in the output template are placeholders.
- Replace every null with a value calculated from the current input image.
- The final output must not contain null values.
- product_x, product_y, and product_scale must be JSON numbers, not strings.

[Output JSON Schema]
{
  "product_analysis": {
    "objects": [],
    "colors": [],
    "camera_angle": "",
    "visual_features": []
  },
  "generation_prompt": {
    "background_prompt": "",
    "negative_prompt": ""
  },
    "layout": {
      "product_position": null,
      "product_x": null,
      "product_y": null,
      "product_scale": null,
      "headline_position": null
    }
}
""".strip()

DEFAULT_POSITIVE = (
    "unoccupied commercial product advertising scene, "
    "realistic supporting surface directly beneath the conditioned product, "
    "matching camera perspective and natural lighting"
)

DEFAULT_NEGATIVE = (
    "person, people, woman, man, child, human, face, portrait, head, body, "
    "hands, mannequin, character, floating product, pasted cutout, harsh outline, "
    "black edge, duplicate main product, extra main product, distorted product, "
    "merged objects, text, logo, watermark"
)




def image_to_data_url(image_path):
    image_path = Path(image_path)

    extension = image_path.suffix.lower().lstrip(".")
    mime_types = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "webp": "webp",
        "gif": "gif",
    }

    if extension not in mime_types:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {extension}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime_types[extension]};base64,{encoded}"


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"응답에서 JSON을 찾지 못했습니다:\n{text}")

        return json.loads(text[start:end + 1])


def build_user_instruction(product_info, direction):
    return f"""
다음 상품 이미지와 상품/매장 정보를 분석하여 광고 생성용 JSON을 작성해라.

광고 방향: {direction}

상품/매장 정보:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

입력 이미지에 여러 상품이 하나의 세트로 촬영되어 있다면,
각 상품을 분리하지 말고 전체 세트를 하나의 광고 상품 구성으로 유지해라.

배경은 상품이 자연스럽게 놓일 실제 지지면과 상업 광고에 적합한 환경을
제공해야 하며 인물, 얼굴, 손, 신체는 절대 포함하지 마라.
""".strip()


def append_prompt(original, required):
    original = str(original or "").strip().strip(",")
    required = required.strip().strip(",")

    if not original:
        return required

    return f"{original}, {required}"


def clamp_float(value, minimum, maximum, default):
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_string_list(value):
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def normalize_prompt_json(data):
    if not isinstance(data, dict):
        raise ValueError("GPT 응답의 최상위 값이 JSON 객체가 아닙니다.")

    for key in PROMPT_SCHEMA_KEYS:
        if key not in data:
            raise ValueError(f"프롬프트 JSON에 필수 키가 없습니다: {key}")

    product_analysis = data.setdefault("product_analysis", {})
    product_analysis["objects"] = normalize_string_list(
        product_analysis.get("objects")
    )
    product_analysis["colors"] = normalize_string_list(
        product_analysis.get("colors")
    )
    product_analysis["visual_features"] = normalize_string_list(
        product_analysis.get("visual_features")
    )
    product_analysis["camera_angle"] = str(
        product_analysis.get("camera_angle", "")
    ).strip()

    ad_copies = normalize_string_list(data.get("ad_copies"))
    data["ad_copies"] = ad_copies[:3]

    while len(data["ad_copies"]) < 3:
        data["ad_copies"].append("")

    generation_prompt = data.setdefault("generation_prompt", {})

    background_prompt = generation_prompt.get("background_prompt", "")
    negative_prompt = generation_prompt.get("negative_prompt", "")

    generation_prompt["background_prompt"] = append_prompt(
        background_prompt,
        DEFAULT_POSITIVE,
    )
    generation_prompt["negative_prompt"] = append_prompt(
        negative_prompt,
        DEFAULT_NEGATIVE,
    )

    layout = data.setdefault("layout", {})
    layout["product_position"] = str(
        layout.get("product_position", "lower_center")
    )
    layout["product_x"] = clamp_float(
        layout.get("product_x"),
        0.15,
        0.85,
        0.50,
    )
    layout["product_y"] = clamp_float(
        layout.get("product_y"),
        0.35,
        0.90,
        0.70,
    )
    layout["product_scale"] = clamp_float(
        layout.get("product_scale"),
        0.25,
        0.60,
        0.44,
    )
    layout["headline_position"] = str(
        layout.get("headline_position", "top_center")
    )

    return data


def run_prompt_generation(
    image_path,
    info_path,
    output_path,
    model="gpt-5.4-nano",
    direction="product_focus",
    detail="low",
):
    image_path = Path(image_path)
    info_path = Path(info_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"상품 이미지를 찾을 수 없습니다: {image_path}"
        )

    if not info_path.exists():
        raise FileNotFoundError(
            f"상품 정보 JSON을 찾을 수 없습니다: {info_path}"
        )

    product_info = json.loads(
        info_path.read_text(encoding="utf-8")
    )
    image_url = image_to_data_url(image_path)

    client = OpenAI()

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_instruction(
                            product_info=product_info,
                            direction=direction,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    result = extract_json(response.output_text)
    result = normalize_prompt_json(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] prompt json saved: {output_path}")
    return output_path
