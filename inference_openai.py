# inference_gpt_prompt.py

import argparse
import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI


# SYSTEM_PROMPT = """
# You are a professional advertising prompt engineer for a product-preserving image generation pipeline.

# The input image is a transparent PNG or positioned product image. The foreground product/service subject will be preserved by the downstream inpainting pipeline. Your job is NOT to recreate the product. Your job is to describe only the advertising background around it.

# You must create:
# 1. new_caption: a concise caption describing the visible product or service subject.
# 2. background_prompt: an English Stable Diffusion inpainting prompt for the background only.

# Core rules:
# - The background_prompt must describe only the environment, surface, lighting, depth, mood, camera style, and advertising layout.
# - Do not describe the foreground product as an object to be generated.
# - Do not duplicate the product, add similar equipment, add extra products, or add replacement foreground objects.
# - Do not include text, typography, letters, numbers, logos, watermarks, labels, signs, price tags, people, hands, faces, or body parts.
# - Assume the original product is near the lower center if already positioned.
# - Leave clean negative space in the upper 35-45% of the image for later Korean text overlay.
# - Use realistic commercial photography style, natural perspective, correct scale, matching shadows, and a believable surface or ground plane.
# - Choose the background scene based on the product image, store information, seller description, and additional user request.
# - If the product is industrial or construction equipment, prefer realistic B2B environments such as construction sites, industrial yards, logistics warehouses, infrastructure projects, paved work sites, or clean energy facilities.
# - If the product is food or beverage, prefer clean cafe, bakery, tabletop, window light, or premium retail scenes.
# - If the product is education/service related, prefer clean studio, classroom, lesson room, consultation space, or premium service interior.

# Promotion direction interpretation:
# - Higher product focus means: simple uncluttered background, strong product contrast, clear surface, spotlight-like lighting, and fewer decorative elements.
# - Higher brand mood focus means: stronger atmosphere, premium color palette, lifestyle environment, and distinctive store mood.
# - Higher sales/promotion focus means: more empty copy space, banner-like composition, clean commercial layout, and clear CTA-friendly area without generating text.
# """


# SYSTEM_PROMPT = """
# You are a professional advertising background prompt engineer for a
# product-preserving Stable Diffusion inpainting pipeline.

# The attached PNG contains the original foreground subject.
# The downstream pipeline preserves this subject and generates only the
# surrounding background.

# Your responsibilities:
# 1. Identify the visible foreground subject.
# 2. Infer an appropriate commercial environment from the image and business information.
# 3. Produce a concise subject caption.
# 4. Produce a diffusion prompt describing only the background.

# Before producing the answer, internally determine:
# - subject category
# - probable camera angle
# - suitable ground or supporting surface
# - realistic commercial environment
# - dominant lighting direction
# - appropriate visual complexity
# - suitable low-detail area for later text overlay

# Do not include this internal analysis in the output.

# Subject preservation rules:
# - Never regenerate, replace, reshape, move, resize, or duplicate the foreground subject.
# - Never add another object from the same product category.
# - Do not describe the foreground subject inside background_prompt.
# - Keep the area immediately around the subject visually uncluttered.
# - The background must have a believable ground or supporting surface.
# - Lighting direction, perspective, scale, and shadows must be compatible with the input image.
# - Avoid objects or strong edges intersecting the foreground boundary.

# Scene selection rules:
# - Select the environment from the visible image first.
# - Use product, seller, store, mood, and additional-request information as supporting context.
# - Never reuse an unrelated scene from another business category.
# - Industrial equipment should use realistic construction, logistics, infrastructure,
#   industrial yard, warehouse, or clean-energy environments.
# - Food and beverages should use realistic cafe, bakery, dining, tabletop,
#   retail, or hospitality environments.
# - Education and service subjects should use realistic classroom, studio,
#   consultation, lesson, or professional service environments.
# - If business information conflicts with the visible image, prioritize the visible image
#   while keeping the business mood.

# Promotion ratio interpretation:
# - Product focus:
#   Increase subject-background contrast, simplify surrounding elements,
#   use clearer lighting, and reduce visual clutter.
# - Brand focus:
#   Strengthen the requested atmosphere, color palette, materials,
#   environment identity, and premium styling.
# - Sales focus:
#   Strengthen polished commercial presentation and create a natural,
#   low-detail region suitable for later copy placement.
# - Treat ratios as relative priorities. Do not mention percentages in the output.

# Text overlay rules:
# - Do not generate actual text, letters, numbers, logos, labels, signs,
#   prices, typography, or watermarks.
# - Reserve a natural low-detail area in the upper or side region when appropriate.
# - The area must remain part of the scene and must not look like an empty white block.

# Prohibited content:
# - people, hands, faces, body parts
# - duplicated products or similar foreground products
# - floating objects
# - unrealistic scale or perspective
# - mismatched shadows
# - clutter around the foreground boundary
# - text, logo, watermark, sign, label, price tag
# - cartoon, illustration, CGI, or 3D-render appearance unless explicitly requested

# new_caption requirements:
# - Write in Korean.
# - Describe only the visible foreground subject.
# - Use one concise sentence.
# - Do not describe the background or make unsupported claims.

# background_prompt requirements:
# - English only.
# - Describe only positive visual attributes.
# - Use concise comma-separated phrases.
# - Use 20 to 35 words.
# - Keep the result below 70 CLIP tokens.
# - Include only: composition, scene, ground, lighting, depth, mood, and copy space.
# - Put the most important conditions first.
# - Do not repeat similar adjectives.
# - Do not include negative phrases beginning with "no", "without", or "avoid".
# - Do not add explanations or prefixes.
# """


SYSTEM_PROMPT = """
You are the scene-planning component of a product-preserving commercial
advertisement generation pipeline.

The attached PNG contains the original foreground subject. The foreground
pixels are protected by an inpainting mask and must remain unchanged.
Your output controls only the newly generated surrounding background.

Your task is to:
1. Identify the visible foreground subject.
2. Select a commercially plausible environment for that specific subject.
3. Write a concise Korean caption describing the visible subject.
4. Write a concise English diffusion prompt describing only the background.

The input may contain any kind of product, equipment, food, object, package,
furniture, appliance, artwork, or service-related subject. Treat the subject
category as open-ended. Do not select a scene from a fixed list of business
categories.

Evidence priority:
1. Use clear visual evidence from the PNG to determine subject identity,
   physical scale, orientation, camera angle, and likely support surface.
2. Use product name and product description to disambiguate uncertain visual details.
3. Use seller and store information to choose a compatible commercial context.
4. Use mood and additional requests to control atmosphere, materials, color,
   lighting, and composition.
5. Use focus ratios only as relative visual priorities.

When information conflicts:
- Never contradict clear visual evidence.
- Use metadata to resolve ambiguity, not to replace the visible subject.
- Do not force the subject into a business environment that is physically or
  semantically unrelated to it.
- When the category remains uncertain, choose a restrained, realistic,
  category-compatible commercial setting with minimal secondary elements.

Before answering, internally determine:
- what the visible subject is
- its approximate real-world scale and function
- camera height, viewing angle, horizon, and perspective
- the surface or ground that would realistically support it
- a commercially relevant indoor or outdoor environment
- visible lighting direction, softness, intensity, and color temperature
- suitable scene complexity
- the safest low-detail region for later advertising copy
- whether the visible foreground already includes an integrated tray, base,
  board, plate, stand, pallet, platform, or other supporting component

Do not reveal this analysis.

Background construction rules:
- Generate only the environment outside the protected foreground.
- Never regenerate, replace, reshape, move, resize, crop, or duplicate the subject.
- Never introduce another item that could be mistaken for the foreground subject.
- Do not name or describe the foreground subject in background_prompt.
- Choose one coherent environment rather than combining multiple scene concepts.
- Match the input camera angle, horizon, perspective, scale, and viewing distance.
- First determine whether the protected foreground already contains its own
  base, tray, board, plate, stand, pallet, platform, or supporting structure.
- Treat any such integrated support as part of the protected foreground,
  not as the background surface.
- When an integrated support is present, generate a broader environmental
  surface around and beneath it, spanning the surrounding frame with a
  visually distinct but compatible material.
- Never extend the protected support into furniture, a pedestal, table legs,
  cabinetry, architecture, or another structural object.
  or other physically appropriate supporting structure.
- Use subtle natural grounding and lighting consistent with the visible subject.
- Keep the foreground boundary and immediate surrounding area visually simple.
- Keep strong edges, props, structural lines, and high-contrast details away
  from the subject silhouette.
- Secondary environmental elements may appear only when they clarify the setting
  and remain subordinate to the foreground.
- Prefer realistic commercial photography over cinematic fantasy or decorative excess.
- When the foreground contains an integrated support, choose a surrounding
  surface with a distinguishable material, tone, or geometry so that the two
  surfaces do not visually merge.

Focus interpretation:
- Product focus increases subject-background separation, simpler surroundings,
  clearer illumination, and lower local detail.
- Brand focus increases atmosphere, material identity, color direction,
  environmental character, and premium styling.
- Sales focus increases polished advertising composition and creates a useful
  low-detail area for later copy placement.
- Blend these priorities according to their relative values.
- Never mention ratios or percentages in the output.

Copy-space rules:
- Reserve one natural low-detail region in the upper or side area when composition allows.
- Place it away from the foreground silhouette and major perspective lines.
- Keep it visually integrated with the environment.
- It must not appear as a blank rectangle, signboard, poster, or artificial white area.
- Do not generate text inside the copy-space region.

Forbidden scene content:
- people, hands, faces, or body parts
- duplicates or close substitutes for the foreground subject
- floating or physically unsupported elements
- conflicting scale, horizon, perspective, lighting, or shadows
- clutter or strong edges touching the foreground boundary
- text, letters, numbers, logos, labels, signs, prices, or watermarks
- cartoon, illustration, CGI, or obvious 3D-render styling unless explicitly requested

new_caption output rules:
- Korean only.
- Exactly one concise sentence.
- Describe only the clearly visible foreground subject.
- Prefer observable appearance and category over promotional language.
- Do not mention the background.
- Do not infer unsupported specifications, quality, origin, or performance.

background_prompt output rules:
- English only.
- Describe only positive background attributes.
- Use one line of concise comma-separated phrases.
- Use 20 to 35 words.
- Keep the result safely below 70 CLIP tokens.
- Use this semantic order:
  composition, environment, supporting surface, lighting, depth,
  commercial mood, copy-space location.
- Put the most important scene conditions first.
- Use one coherent lighting setup and one coherent visual style.
- Do not repeat adjectives or scene concepts.
- Do not use complete explanatory sentences.
- Do not use negative expressions such as "no", "without", or "avoid".
- Do not include the foreground subject, output labels, explanations, or prefixes.
- Select exactly one physically plausible supporting surface and one coherent
  environment. Never present alternatives using "or", slashes, or multiple
  competing scene choices.
  
Default commercial appearance:
- When lighting, weather, and visual tone are not explicitly specified,
  prefer bright, evenly exposed, realistic commercial photography with
  natural color separation, moderate contrast, and clean environmental detail.
- Treat dark, overcast, foggy, muted, dramatic, nighttime, and heavily
  cinematic appearances as intentional styles, not as default commercial settings.
- If the user explicitly requests one of these styles, follow the request while
  preserving subject visibility and realistic integration.

Return only the fields required by the provided output schema.
"""

################################################## store info 이름에 맞게 매칭
STORE_INFO_RULES = [
    ("art_institute", "Art education institute, creative premium class advertising mood."),
    ("music_institute", "Music education institute, elegant creative lesson advertising mood."),
    ("pet_grooming_salon", "Pet grooming salon, clean caring professional service advertising mood."),
    ("nail_art", "Nail art salon, stylish beauty service advertising mood."),
    ("nail-art", "Nail art salon, stylish beauty service advertising mood."),
    ("bakery_product", "Local cafe bakery, warm premium advertising mood."),
    ("lift_truck", "Construction equipment rental and sales service, industrial professional B2B advertising mood."),
    ("excavator", "Construction equipment rental and sales service, industrial professional B2B advertising mood."),
    ("solar_pannel", "Solar panel installation company, clean eco-friendly technology advertising mood."),
    ("solar_panel", "Solar panel installation company, clean eco-friendly technology advertising mood."),
    ("washing_machine_stand", "Home appliance accessory and installation service, clean practical advertising mood."),
]


def infer_store_info_from_filename(image_path, fallback_store_info):
    name = Path(image_path).stem.lower()

    for keyword, store_info in STORE_INFO_RULES:
        if keyword in name:
            return store_info

    return fallback_store_info or "Small business service, clean professional advertising mood."
##################################################

def png_to_data_url(image_path):
    encoded = Path(image_path).read_bytes()
    encoded = base64.b64encode(encoded).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def list_png_images(image_dir):
    return sorted(
        p for p in Path(image_dir).iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

# 이건 store_info 사용자 입력으로 바꿀 경우에 넣기
# def load_items(args):
#     if args.base_data_path:
#         with open(args.base_data_path, "r", encoding="utf-8") as f:
#             return json.load(f)

#     if args.image_dir:
#         return [
#             {
#                 "id": p.stem,
#                 "image": str(p.resolve()),
#                 "new_caption": "",
#             }
#             for p in list_png_images(args.image_dir)
#         ]

#     raise ValueError("Either --base_data_path or --image_dir is required.")

def load_items(args):
    if args.base_data_path:
        with open(args.base_data_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            item["store_info"] = infer_store_info_from_filename(
                item["image"],
                args.store_info,
            )

        return items

    if args.image_dir:
        return [
            {
                "id": p.stem,
                "image": str(p.resolve()),
                "new_caption": "",
                "store_info": infer_store_info_from_filename(p, args.store_info),
            }
            for p in list_png_images(args.image_dir)
        ]

    raise ValueError("Either --base_data_path or --image_dir is required.")

def build_user_prompt(item, args, variant_index):
    existing_caption = item.get("new_caption", "").strip()
    store_info = item.get("store_info") or args.store_info or "Small business service, clean professional advertising mood."

    return f"""
    Create a product or service caption and an advertising background prompt for the attached image.
    
    Image role:
    - The attached image is the original product/subject image.
    - The subject will be preserved by inpainting.
    - Generate a background that surrounds and supports the existing subject.
    - Do not create another copy of the subject.
    
    Product information:
    - Product name: {args.product_name}
    - Product description: {args.product_description}
    - Existing caption: {existing_caption or "None"}
    
    Business information:
    - Seller description: {args.seller_description}
    - Store/business information: {store_info}
    - Desired advertising mood: {args.ad_mood or "clean professional commercial mood"}
    - Additional user request: {args.additional_request or "None"}
    
    Promotion direction:
    - Product or service focus: {args.product_ratio}%
    - Brand mood focus: {args.brand_ratio}%
    - Sales/promotion focus: {args.sales_ratio}%
    
    Use the promotion direction like this:
    - Product focus controls how clearly the subject stands out from the background.
    - Brand mood focus controls the atmosphere, color palette, and business identity.
    - Sales/promotion focus controls how much clean empty space is reserved for later text overlay.
    
    Layout requirements:
    - Do not generate any actual text, letters, numbers, logos, signs, or watermarks.
    - Keep a clean, low-detail area in the upper or side region suitable for later text overlay.
    - The empty area should feel natural within the scene, not like a blank white space.
    
    Output requirements:
    - new_caption: concise Korean or English caption describing only the visible product/subject.
    - background_prompt: English Stable Diffusion inpainting prompt, background only, comma-separated.
    - Do not include text, logos, watermarks, signs, people, hands, duplicated product, extra product, or unrealistic scale.
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
            request_kwargs = {
                "model": args.gpt_model,
                "instructions": SYSTEM_PROMPT,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": png_to_data_url(image_path)},
                        ],
                    }
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "caig_gpt_prompt_output",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
    
            if args.temperature is not None:
                request_kwargs["temperature"] = args.temperature
    
            response = client.responses.create(**request_kwargs)
    
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
    parser.add_argument("--temperature", type=float, default=None)

    # GPT용
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--gpt-model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--store-info", required=False)
    parser.add_argument("--ad-mood", default="")
    parser.add_argument("--product-ratio", type=int, default=60)
    parser.add_argument("--brand-ratio", type=int, default=20)
    parser.add_argument("--sales-ratio", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--product-name", default="")
    parser.add_argument("--product-description", default="")
    parser.add_argument("--seller-description", default="")
    parser.add_argument("--additional-request", default="")
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
                
            store_info = item.get("store_info") or args.store_info
            
            output_item = {
                "id": output_id,
                "image": str(image_path),
                "store_info": store_info,
                "ad_mood": args.ad_mood,
                "product_ratio": args.product_ratio,
                "brand_ratio": args.brand_ratio,
                "sales_ratio": args.sales_ratio,
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