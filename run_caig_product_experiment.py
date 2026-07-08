import argparse
import json
import shutil
import subprocess
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def replace_image_path(obj, new_path):
    if isinstance(obj, dict):
        updated = {}
        for k, v in obj.items():
            key = k.lower()
            if isinstance(v, str) and (
                "image" in key or Path(v).suffix.lower() in IMAGE_EXTS
            ):
                updated[k] = new_path
            else:
                updated[k] = replace_image_path(v, new_path)
        return updated

    if isinstance(obj, list):
        return [replace_image_path(item, new_path) for item in obj]

    return obj


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="input product image path")
    parser.add_argument("--name", default=None, help="experiment name")
    parser.add_argument("--template-json", default="tiny_dataset/tiny.json")
    parser.add_argument("--prompt-model", default="weights/llava-v1.6-vicuna-7b-pretrain")
    parser.add_argument("--controlnet-model", default="lllyasviel/control_v11p_sd15_canny")
    parser.add_argument("--base-model", default="digiplay/majicMIX_realistic_v7")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generate-nums", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampler", default="Euler a")
    args = parser.parse_args()

    root = Path.cwd()
    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    exp_name = args.name or image_path.stem
    image_dst_dir = root / "tiny_dataset" / "images"
    image_dst_dir.mkdir(parents=True, exist_ok=True)

    copied_image = image_dst_dir / f"{exp_name}{image_path.suffix.lower()}"
    shutil.copy2(image_path, copied_image)

    relative_image_path = copied_image.relative_to(root).as_posix()

    with open(args.template_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = replace_image_path(data, relative_image_path)

    dataset_path = root / "tiny_dataset" / f"{exp_name}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    output_root = root / "output_compare" / exp_name
    prompt_path = output_root / "prompt" / "output_prompt.json"
    image_output_dir = output_root / "images"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    run([
        "accelerate", "launch", "inference_llava.py",
        "--model-path", str((root / args.prompt_model).resolve()),
        "--output_data_path", str(prompt_path),
        "--generate_nums", str(args.generate_nums),
        "--base_data_path", str(dataset_path),
        "--temperature", str(args.temperature),
    ])

    integration_suffix = (
    ", cohesive commercial product advertising photo, "
    "the product naturally placed in the scene, "
    "realistic contact shadow, matched lighting and perspective, "
    "consistent color tone, not pasted, not floating, not cutout"
    )

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    
    def add_integration_suffix(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and "prompt" in k.lower():
                    obj[k] = v + integration_suffix
                else:
                    add_integration_suffix(v)
        elif isinstance(obj, list):
            for item in obj:
                add_integration_suffix(item)
    
    add_integration_suffix(prompt_data)
    
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=2)

    run([
        "accelerate", "launch", "sample_llava.py",
        "--batch_size", str(args.batch_size),
        "--base_model_path", args.base_model,
        "--save_path", str(image_output_dir),
        "--controlnet_model_path", args.controlnet_model,
        "--num_inference_steps", str(args.steps),
        "--sampler_name", args.sampler,
        "--data_path", str(prompt_path),
    ])

    print("\n[DONE]")
    print(f"dataset json: {dataset_path}")
    print(f"prompt json : {prompt_path}")
    print(f"outputs     : {image_output_dir}")


if __name__ == "__main__":
    main()