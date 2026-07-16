import argparse
from pathlib import Path
from PIL import Image


def crop_to_alpha(image):
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        return image

    return image.crop(bbox)


def place_product_bottom(input_path, output_path, width, height, scale, bottom_margin):
    product = Image.open(input_path).convert("RGBA")
    product = crop_to_alpha(product)

    max_w = int(width * scale)
    max_h = int(height * scale)

    product.thumbnail((max_w, max_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    pw, ph = product.size

    x = (width - pw) // 2
    y = height - ph - bottom_margin

    if y < 0:
        y = 0

    canvas.alpha_composite(product, (x, y))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--bottom-margin", type=int, default=64)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))

    count = 0

    for image_path in image_paths:
        output_path = output_dir / image_path.name

        place_product_bottom(
            input_path=image_path,
            output_path=output_path,
            width=args.width,
            height=args.height,
            scale=args.scale,
            bottom_margin=args.bottom_margin,
        )

        print(f"saved: {output_path}")
        count += 1

    print(f"done: {count} files")


if __name__ == "__main__":
    main()