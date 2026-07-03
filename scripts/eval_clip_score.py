import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_WORDS = {"mask", "canny", "edge", "concat", "control", "condition"}


def list_images(image_dir):
    image_dir = Path(image_dir)
    paths = []

    for p in image_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue

        name = p.stem.lower()
        if any(word in name for word in SKIP_WORDS):
            continue

        paths.append(p)

    return sorted(paths)


def build_eval_text(item, store_info):
    parts = []

    if store_info:
        parts.append(store_info)

    if item.get("new_caption"):
        parts.append("Product: " + item["new_caption"])

    if item.get("answer"):
        parts.append("Background prompt: " + item["answer"])

    parts.append("advertising image, visually appealing, no text")

    return ". ".join(parts)


def clip_score(model, processor, image_path, text, device):
    image = Image.open(image_path).convert("RGB")

    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        image_feat = model.get_image_features(**image_inputs)
        text_feat = model.get_text_features(**text_inputs)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        score = image_feat @ text_feat.T

    return float(score.item())


def find_candidate_images(all_images, item_id):
    item_id = str(item_id).lower()

    matched = []
    for p in all_images:
        stem = p.stem.lower()

        if stem == item_id:
            matched.append(p)
        elif stem.startswith(item_id + "_"):
            matched.append(p)
        elif item_id in stem:
            matched.append(p)

    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--store-info", default="")
    parser.add_argument("--output-json", default="output_demo/eval_clip_score/clip_scores.json")
    parser.add_argument("--output-csv", default="output_demo/eval_clip_score/clip_scores.csv")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = json.load(open(args.prompt_json, "r", encoding="utf-8"))
    all_images = list_images(args.image_dir)

    print(f"device: {device}")
    print(f"prompt items: {len(data)}")
    print(f"candidate images: {len(all_images)}")

    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)

    results = []

    for idx, item in enumerate(data):
        item_id = item.get("id", idx)
        eval_text = build_eval_text(item, args.store_info)
        candidates = find_candidate_images(all_images, item_id)

        if not candidates:
            print(f"warning: no image matched for id={item_id}")
            continue

        for image_path in candidates:
            score = clip_score(model, processor, image_path, eval_text, device)

            row = {
                "id": item_id,
                "image": str(image_path),
                "clip_score": score,
                "eval_text": eval_text,
            }

            results.append(row)
            print(f"{item_id}: {score:.4f} | {image_path}")

    results.sort(key=lambda x: x["clip_score"], reverse=True)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image", "clip_score", "eval_text"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")

    if results:
        print(f"best: {results[0]['clip_score']:.4f} | {results[0]['image']}")


if __name__ == "__main__":
    main()