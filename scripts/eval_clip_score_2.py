import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_WORDS = {
    "mask",
    "canny",
    "edge",
    "concat",
    "control",
    "condition",
}


def list_images(image_dir):
    image_dir = Path(image_dir)
    paths = []

    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue

        name = path.stem.lower()
        if any(word in name for word in SKIP_WORDS):
            continue

        paths.append(path)

    return sorted(paths)


def list_cutout_images(cutout_dir):
    cutout_dir = Path(cutout_dir)
    paths = []

    for path in cutout_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        if path.suffix.lower() != ".png":
            continue

        paths.append(path)

    return sorted(paths)


def text_value(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " | ".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    return str(value).strip()


def load_existing_json(output_json):
    if not output_json.exists():
        return []

    try:
        with open(output_json, "r", encoding="utf-8") as file:
            existing_results = json.load(file)

        if not isinstance(existing_results, list):
            return []

        return existing_results

    except (json.JSONDecodeError, OSError):
        return []


def build_background_eval_text(item, store_info):
    background_prompt = text_value(item.get("answer"))

    parts = [
        store_info,
        f"Background prompt: {background_prompt}",
        "advertising background, visually appealing, no text",
    ]

    return ". ".join(part for part in parts if part)


def find_candidate_images(all_images, item_id):
    item_id = str(item_id).lower()
    matched = []

    for image_path in all_images:
        stem = image_path.stem.lower()
        if item_id in stem:
            matched.append(image_path)

    if matched:
        return matched

    return all_images


def find_cutout_image(cutout_images, item_id):
    item_id = str(item_id).lower()

    for cutout_path in cutout_images:
        if item_id in cutout_path.stem.lower():
            return cutout_path

    if len(cutout_images) == 1:
        return cutout_images[0]

    return None


def create_background_only_image(
    final_image_path,
    cutout_image_path,
    alpha_threshold,
    fill_value,
):
    final_image = Image.open(final_image_path).convert("RGB")
    cutout_image = Image.open(cutout_image_path).convert("RGBA")

    if cutout_image.size != final_image.size:
        cutout_image = cutout_image.resize(final_image.size)

    background_only = final_image.copy()
    pixels = background_only.load()
    alpha = cutout_image.getchannel("A")

    for y in range(background_only.height):
        for x in range(background_only.width):
            if alpha.getpixel((x, y)) >= alpha_threshold:
                pixels[x, y] = (
                    fill_value,
                    fill_value,
                    fill_value,
                )

    return background_only


def clip_score(model, processor, image, text, device):
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(
            dim=-1,
            keepdim=True,
        )

        text_embeds = text_embeds / text_embeds.norm(
            dim=-1,
            keepdim=True,
        )

        score = torch.matmul(
            image_embeds,
            text_embeds.T,
        ).item()

    return float(score)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--cutout-dir", required=True)
    parser.add_argument("--store-info", default="")

    parser.add_argument(
        "--output-json",
        default=(
            "output_demo/eval_clip_score_2/"
            "clip_background_scores.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "output_demo/eval_clip_score_2/"
            "clip_background_scores.csv"
        ),
    )

    parser.add_argument(
        "--masked-image-dir",
        default="",
    )

    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--mask-fill-value",
        type=int,
        default=127,
    )

    parser.add_argument("--rembg-model", default="isnet-general-use")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llava-model", default="")
    parser.add_argument("--diffusion-model", default="")
    parser.add_argument("--controlnet-model", default="")

    args = parser.parse_args()

    if not 0 <= args.alpha_threshold <= 255:
        raise ValueError("--alpha-threshold must be between 0 and 255")

    if not 0 <= args.mask_fill_value <= 255:
        raise ValueError("--mask-fill-value must be between 0 and 255")

    prompt_json = Path(args.prompt_json)
    image_dir = Path(args.image_dir)
    cutout_dir = Path(args.cutout_dir)

    if not prompt_json.exists():
        raise FileNotFoundError(f"prompt-json not found: {prompt_json}")

    if not image_dir.exists():
        raise FileNotFoundError(f"image-dir not found: {image_dir}")

    if not cutout_dir.exists():
        raise FileNotFoundError(f"cutout-dir not found: {cutout_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(prompt_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("prompt-json root value must be a list")

    all_images = list_images(image_dir)
    cutout_images = list_cutout_images(cutout_dir)

    print(f"device: {device}")
    print(f"prompt items: {len(data)}")
    print(f"candidate images: {len(all_images)}")
    print(f"cutout images: {len(cutout_images)}")

    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    masked_image_dir = None

    if args.masked_image_dir:
        masked_image_dir = Path(args.masked_image_dir)
        masked_image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    results = []

    for idx, item in enumerate(data):
        item_id = item.get("id", idx)

        gpt_product_caption = text_value(item.get("new_caption"))
        llava_question = text_value(item.get("question"))
        llava_background_prompt = text_value(item.get("answer"))

        eval_text = build_background_eval_text(
            item,
            args.store_info,
        )

        candidates = find_candidate_images(
            all_images,
            item_id,
        )

        if not candidates:
            print(f"warning: no generated image matched for id={item_id}")
            continue

        for image_path in candidates:
            cutout_path = find_cutout_image(
                cutout_images,
                item_id,
            )

            if cutout_path is None:
                print(
                    "warning: no cutout image matched: "
                    f"id={item_id}, generated={image_path}"
                )
                continue

            background_only_image = create_background_only_image(
                final_image_path=image_path,
                cutout_image_path=cutout_path,
                alpha_threshold=args.alpha_threshold,
                fill_value=args.mask_fill_value,
            )

            masked_image_path = ""

            if masked_image_dir is not None:
                masked_output_path = (
                    masked_image_dir
                    / f"{image_path.stem}_background_only.png"
                )

                background_only_image.save(masked_output_path, "PNG")
                masked_image_path = str(masked_output_path)

            score = clip_score(
                model=model,
                processor=processor,
                image=background_only_image,
                text=eval_text,
                device=device,
            )

            row = {
                "id": item_id,
                "gpt_product_caption": gpt_product_caption,
                "llava_question": llava_question,
                "llava_background_prompt": llava_background_prompt,
                "image": str(image_path),
                "cutout_image": str(cutout_path),
                "masked_image": masked_image_path,
                "clip_score": score,
                "eval_text": eval_text,
                "alpha_threshold": args.alpha_threshold,
                "mask_fill_value": args.mask_fill_value,
                "rembg_model": args.rembg_model,
                "llm_model": args.llm_model,
                "llava_model": args.llava_model,
                "diffusion_model": args.diffusion_model,
                "controlnet_model": args.controlnet_model,
                "clip_model": args.model,
            }

            results.append(row)

            print(
                f"{item_id}: {score:.4f} | "
                f"image={image_path} | "
                f"cutout={cutout_path}"
            )

    results.sort(
        key=lambda row: row["clip_score"],
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "gpt_product_caption",
        "llava_question",
        "llava_background_prompt",
        "image",
        "cutout_image",
        "masked_image",
        "clip_score",
        "eval_text",
        "alpha_threshold",
        "mask_fill_value",
        "rembg_model",
        "llm_model",
        "llava_model",
        "diffusion_model",
        "controlnet_model",
        "clip_model",
    ]

    existing_results = load_existing_json(output_json)
    all_results = existing_results + results

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    existing_csv_results = []

    if output_csv.exists() and output_csv.stat().st_size > 0:
        try:
            with open(
                output_csv,
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                existing_csv_results = list(csv.DictReader(file))
        except OSError:
            existing_csv_results = []

    all_csv_results = existing_csv_results + results

    with open(
        output_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in all_csv_results:
            writer.writerow({
                field: row.get(field, "")
                for field in fieldnames
            })

    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")
    print(f"new results: {len(results)}")

    if results:
        print(
            f"best: {results[0]['clip_score']:.4f} | "
            f"{results[0]['image']}"
        )


if __name__ == "__main__":
    main()