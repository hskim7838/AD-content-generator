# inference_gpt_prompt.py

import argparse
import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """
You are an advertising prompt generator for a product-preserving image generation pipeline.

The input product PNG has transparent background and the product will be preserved.
Create:
1. A concise product caption.
2. A Stable Diffusion background prompt for advertising image generation.

Rules:
- Describe only background, surface, mood, lighting, and commercial scene.
- Do not create or duplicate the product.
- Do not include text, typography, letters, logos, watermarks, price tags, labels, people, or hands.
- The background must complement and highlight the original product.
- The background prompt must be short, comma-separated, and written in English.
"""


def png_to_data_url(image_path):
    encoded = Path(image_path).read_bytes()
    encoded = base64.b64encode(encoded).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def list_png_images(image_dir):
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )


def load_items(args):
    if args.base_data_path:
        with open(args.base_data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if args.image_dir:
        return [
            {
                "id": p.stem,
                "image": str(p.resolve()),
                "new_caption": "",
            }
            for p in list_png_images(args.image_dir)
        ]

    raise ValueError("Either --base_data_path or --image_dir is required.")


def build_user_prompt(item, args, variant_index):
    caption = item.get("new_caption", "")

    return f"""
Create a product caption and an advertising background prompt.

Store information:
{args.store_info}

Promotion direction:
- Product focus: {args.product_ratio}%
- Brand mood focus: {args.brand_ratio}%
- Sales/promotion focus: {args.sales_ratio}%

Existing product caption:
{caption}

Requirements:
- Product caption: describe only the visible product.
- Background prompt: Stable Diffusion inpainting prompt.
- Leave clean space for later text overlay.
- Do not include text, logo, watermark, people, hands, or duplicated product.
- Return valid JSON only.

Variant index:
{variant_index}
""".strip()


def call_gpt(client, item, args, variant_index):
    image_path = Path(item["image"])
    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    user_prompt = build_user_prompt(item, args, variant_index)

    schema = {
        "type": "object",
        "properties": {
            "new_caption": {
                "type": "string"
            },
            "background_prompt": {
                "type": "string"
            }
        },
        "required": ["new_caption", "background_prompt"],
        "additionalProperties": False
    }

    last_error = None

    for attempt in range(args.max_retries):
        try:
            response = client.responses.create(
                model=args.gpt_model,
                instructions=SYSTEM_PROMPT,
                temperature=args.temperature,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": png_to_data_url(image_path)},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "caig_gpt_prompt_output",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )

            parsed = json.loads(response.output_text)

            return {
                "question": user_prompt,
                "new_caption": parsed["new_caption"].strip(),
                "answer": parsed["background_prompt"].strip(),
            }

        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"[retry {attempt + 1}/{args.max_retries}] {image_path} | {e}")
            time.sleep(wait)

    raise last_error


def main():
    parser = argparse.ArgumentParser()

    # CAIG inference_llava.py 호환용
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output_data_path", required=True)
    parser.add_argument("--generate_nums", type=int, default=1)
    parser.add_argument("--base_data_path", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    # GPT용
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--gpt-model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--store-info", required=True)
    parser.add_argument("--product-ratio", type=int, default=60)
    parser.add_argument("--brand-ratio", type=int, default=20)
    parser.add_argument("--sales-ratio", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)

    args = parser.parse_args()

    client = OpenAI()
    base_items = load_items(args)

    results = []

    for item in base_items:
        for variant_index in range(args.generate_nums):
            gpt_result = call_gpt(client, item, args, variant_index)

            original_id = str(item.get("id", Path(item["image"]).stem))
            output_id = (
                f"{original_id}_{variant_index}"
                if args.generate_nums > 1
                else original_id
            )

            image_path = Path(item["image"])
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            output_item = {
                "id": output_id,
                "image": str(image_path),
                "new_caption": gpt_result["new_caption"],
                "question": gpt_result["question"],
                "answer": gpt_result["answer"],
            }

            results.append(output_item)
            print(json.dumps(output_item, ensure_ascii=False, indent=2))

    output_path = Path(args.output_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_path}")
    print(f"count: {len(results)}")


if __name__ == "__main__":
    main()