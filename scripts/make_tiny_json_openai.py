import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path

from openai import OpenAI


client = OpenAI()


def slugify(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    return stem.strip("_") or "item"


def image_to_data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_caption(image_path: Path, store_info: str, model: str) -> str:
    prompt = f"""
Create a concise English product caption for CAIG advertising image generation.

Rules:
- Describe only the visible product.
- Include product type, color, material, style, use case, and selling appeal if visible.
- Do not describe the background.
- Do not invent brand names, prices, discount rates, or claims.
- Keep it under 35 words.
- Return JSON only.

Store information:
{store_info}
"""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                    },
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "caig_caption_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "new_caption": {
                            "type": "string"
                        }
                    },
                    "required": ["new_caption"],
                    "additionalProperties": False,
                },
            }
        },
    )

    data = json.loads(response.output_text)
    return data["new_caption"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", default="tiny_dataset/tiny_auto.json")
    parser.add_argument("--path-prefix", default=None)
    parser.add_argument("--store-info", default="")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_path = Path(args.out)

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".png"]
    )

    items = []

    for image_path in image_paths:
        json_image_path = (
            f"{args.path_prefix}/{image_path.name}"
            if args.path_prefix
            else str(image_path)
        )

        item = {
            "id": slugify(image_path.name),
            "image": json_image_path,
            "new_caption": build_caption(
                image_path=image_path,
                store_info=args.store_info,
                model=args.model,
            ),
        }

        print(json.dumps(item, ensure_ascii=False, indent=2))
        items.append(item)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {out_path}")
    print(f"count: {len(items)}")


if __name__ == "__main__":
    main()