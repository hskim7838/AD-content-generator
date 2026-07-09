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
    '''
    for p in image_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue

        name = p.stem.lower()
        if any(word in name for word in SKIP_WORDS):
            continue
    '''

    # 모델 정보 저장 인자를 추가하면서 기존에 생성된 파일에도 해당 정보가 반영되도록 체크포인트 저장 로직을 다음과 같이 수정했습니다.
    for p in image_dir.rglob("*"):
        if not p.is_file():
            continue

        # Jupyter 자동 백업 폴더 제외
        if ".ipynb_checkpoints" in p.parts:
            continue

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

    # 모델에 대한 정보 저장을 위해 아래와 같은 인자를 추가했습니다.
    parser.add_argument("--rembg-model", default="u2net")   # 누끼따는 모델
    parser.add_argument("--llm-model", default="")          # tiny_auto.json 생성 모델
    parser.add_argument("--llava-model", default="")        # 이미지 해석 모델
    parser.add_argument("--diffusion-model", default="")    # diffusion 모델
    parser.add_argument("--controlnet-model", default="")   # controlnet 모델
    
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
                # 모델에 대한 정보 저장을 위해 아래와 같은 인자를 추가했습니다.
                "rembg_model": args.rembg_model,
                "llm_model": args.llm_model,
                "llava_model": args.llava_model,
                "diffusion_model": args.diffusion_model,
                "controlnet_model": args.controlnet_model,
                "clip_model": args.model,
                }

            results.append(row)
            print(f"{item_id}: {score:.4f} | {image_path}")

    results.sort(key=lambda x: x["clip_score"], reverse=True)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    '''
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image", "clip_score", "eval_text",
                        # 모델에 대한 정보 저장을 위해 아래와 같은 인자를 추가했습니다.
                        "rembg_model",
                        "llm_model",
                        "llava_model",
                        "diffusion_model",
                        "controlnet_model",
                        "clip_model",
                       ],
        )
        writer.writeheader()
        writer.writerows(results)
    '''

    # 모델 정보 저장 인자를 추가함에 따라 이전 실행 결과는 유지하면서 새로 생성된 평가 결과만 추가되도록 JSON과 CSV 저장 방식을 누적 저장 방식으로 수정했습니다.
    fieldnames = [
    "id",
    "image",
    "clip_score",
    "eval_text",
    "rembg_model",
    "llm_model",
    "llava_model",
    "diffusion_model",
    "controlnet_model",
    "clip_model",
    ]

    # -------------------------
    # JSON 누적 저장
    # -------------------------
    if output_json.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
    
            if not isinstance(existing_results, list):
                existing_results = []
    
        except (json.JSONDecodeError, OSError):
            existing_results = []
    else:
        existing_results = []
    
    all_results = existing_results + results
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    
    # -------------------------
    # CSV 누적 저장
    # -------------------------
    csv_exists = output_csv.exists() and output_csv.stat().st_size > 0
    
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
    
        if not csv_exists:
            writer.writeheader()
    
        writer.writerows(results)
    
        print(f"saved: {output_json}")
        print(f"saved: {output_csv}")
    
        if results:
            print(f"best: {results[0]['clip_score']:.4f} | {results[0]['image']}")


if __name__ == "__main__":
    main()