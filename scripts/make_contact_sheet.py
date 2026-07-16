import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_font(size):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def fit_image(image, width, height, background):
    image = image.convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), background)

    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))

    return canvas


def shorten_label(text, max_length=42):
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def main():
    parser = argparse.ArgumentParser(
        description="Combine generated images into one contact sheet."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--title", default="Generated image comparison")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=300)
    parser.add_argument("--cell-height", type=int, default=480)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--title-font-size", type=int, default=28)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = []

    for path in input_dir.glob("*.png"):
        if path.resolve() == output_path:
            continue

        if path.name.startswith("contact_sheet"):
            continue

        if args.prefix and not path.name.startswith(args.prefix):
            continue

        image_paths.append(path)

    image_paths.sort(key=natural_sort_key)

    if not image_paths:
        raise ValueError(
            f"No matching PNG images found in {input_dir}"
        )

    columns = max(1, args.columns)
    rows = math.ceil(len(image_paths) / columns)

    background = (245, 245, 245)
    cell_background = (255, 255, 255)
    border_color = (210, 210, 210)
    text_color = (30, 30, 30)

    label_height = args.font_size + 26
    title_height = args.title_font_size + 36

    cell_total_width = args.cell_width + args.gap
    cell_total_height = args.cell_height + label_height + args.gap

    sheet_width = (
        columns * cell_total_width + args.gap
    )
    sheet_height = (
        title_height
        + rows * cell_total_height
        + args.gap
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        background,
    )
    draw = ImageDraw.Draw(sheet)

    title_font = load_font(args.title_font_size)
    label_font = load_font(args.font_size)

    draw.text(
        (args.gap, 16),
        f"{args.title} ({len(image_paths)} images)",
        fill=text_color,
        font=title_font,
    )

    for index, image_path in enumerate(image_paths):
        row = index // columns
        column = index % columns

        x = args.gap + column * cell_total_width
        y = title_height + row * cell_total_height

        with Image.open(image_path) as image:
            fitted = fit_image(
                image,
                args.cell_width,
                args.cell_height,
                cell_background,
            )

        fitted = ImageOps.expand(
            fitted,
            border=1,
            fill=border_color,
        )

        sheet.paste(fitted, (x, y))

        label = shorten_label(image_path.stem)

        label_bbox = draw.textbbox(
            (0, 0),
            label,
            font=label_font,
        )
        label_width = label_bbox[2] - label_bbox[0]

        label_x = x + max(
            0,
            (args.cell_width - label_width) // 2,
        )
        label_y = y + args.cell_height + 10

        draw.text(
            (label_x, label_y),
            label,
            fill=text_color,
            font=label_font,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    sheet.save(output_path, quality=95)

    print(f"saved: {output_path}")
    print(f"images: {len(image_paths)}")
    print(f"grid: {columns} columns x {rows} rows")
    print(f"size: {sheet_width} x {sheet_height}")


if __name__ == "__main__":
    main()