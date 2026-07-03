import argparse
from pathlib import Path

from PIL import Image, ImageOps
from rembg import new_session, remove


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(input_dir: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def remove_background(image_path: Path, output_path: Path, session, max_side: int | None):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGBA")

    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    result = remove(image, session=session)

    if result.mode != "RGBA":
        result = result.convert("RGBA")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="isnet-general-use")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-side", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {input_dir}")

    session = new_session(args.model)
    image_paths = collect_images(input_dir, args.recursive)

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    for image_path in image_paths:
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix(".png")

        if output_path.exists() and not args.force:
            print(f"skip: {output_path}")
            continue

        print(f"remove bg: {image_path} -> {output_path}")
        remove_background(
            image_path=image_path,
            output_path=output_path,
            session=session,
            max_side=args.max_side,
        )

    print(f"done: {len(image_paths)} files")


if __name__ == "__main__":
    main()