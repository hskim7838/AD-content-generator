#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTHONWARNINGS="ignore:.*copying from a non-meta parameter.*:UserWarning"

# # ===== construction paths =====
# INPUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/construction"
# CUTOUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/construction_cutout"
# OUTPUT_PATH="./output_demo"

# POSITIONED_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/construction_positioned"
# CANVAS_WIDTH=512
# CANVAS_HEIGHT=768
# OBJECT_SCALE=0.78
# BOTTOM_MARGIN=128 #원래 64


# ===== bakery paths =====
INPUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images"
CUTOUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images_cutout_sample"
OUTPUT_PATH="./output_demo"

POSITIONED_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images_positioned"
CANVAS_WIDTH=512
CANVAS_HEIGHT=768
OBJECT_SCALE=0.78
BOTTOM_MARGIN=1 #원래 64

# ===== CAIG settings =====
prompt_nums=1
controlnet_model_path="lllyasviel/control_v11p_sd15_canny"
base_model_path="digiplay/majicMIX_realistic_v7"

# ===== OpenAI settings =====
OPENAI_ENV="secrets/openai.env"
# STORE_INFO="Local cafe bakery, fresh pastries and iced americano"
# AD_MOOD="warm, cozy, premium, appetizing, relaxed cafe atmosphere"

# ===== user inputs =====
#     # ==== constrction ====
# PRODUCT_NAME="사진 속 건설 및 산업 장비"

# PRODUCT_DESCRIPTION="이미지에 보이는 단일 건설 장비 또는 산업 설비. 장비의 종류와 외형은 첨부 이미지에서 정확하게 판단하며, 다른 장비를 추가하지 않음"

# SELLER_DESCRIPTION="건설 현장과 산업 작업에 필요한 중장비 렌탈, 산업 설비 공급 및 전문 장비 상담을 제공하는 업체"

# # STORE_INFO="건설 장비 렌탈 및 산업 설비 전문 서비스, 신뢰감 있고 안전하며 현대적인 B2B 광고 분위기"
# STORE_INFO="서울과 경기 지역을 중심으로 굴착기, 크레인, 지게차 등 중장비 렌탈과 현장 맞춤 장비 상담을 제공하는 전문 업체"

# ADDITIONAL_REQUEST="Identify the exact equipment in the image and create a matching vertical B2B background. Preserve the original subject near the lower center. Use a realistic construction site, industrial yard, warehouse, infrastructure project, or clean-energy facility. Include believable ground, matching light and contact shadows, with a natural low-detail upper area. Do not add other equipment."

    # ==== bakery ====
PRODUCT_NAME="수제 베이커리와 아이스커피 세트"

PRODUCT_DESCRIPTION="나무 서빙 트레이 위에 갓 구운 빵, 쿠키, 페이스트리와 아이스커피가 함께 놓인 베이커리 카페 세트"

SELLER_DESCRIPTION="매일 아침 신선한 재료로 빵과 디저트를 직접 굽고 커피를 함께 제공하는 따뜻한 분위기의 동네 베이커리 카페"

STORE_INFO="수제 베이커리와 커피를 판매하는 아늑한 로컬 카페, 따뜻하고 신선하며 프리미엄한 식음료 광고 분위기"

ADDITIONAL_REQUEST="vertical bakery advertisement, cozy cafe interior, 트레이를 책상으로 인식하지 말 것"

PRODUCT_RATIO=70
BRAND_RATIO=20
SALES_RATIO=10

mkdir -p "$CUTOUT_DIR"
mkdir -p "$OUTPUT_PATH"

# 1. Load OpenAI key
source "$OPENAI_ENV"


# 2. Remove background
./rembg/bin/python scripts/remove_bg_rembg.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$CUTOUT_DIR"

# 3. Generate caption + diffusion prompt with OpenAI API
Prompt_Model_Output_Path="$OUTPUT_PATH/bakery/0713/no_neg.json"

rm -f "$Prompt_Model_Output_Path"

# 3. Place product on vertical transparent canvas
mkdir -p "$POSITIONED_DIR"

./caig/bin/python3 scripts/place_product_bottom_batch.py \
  --input-dir "$CUTOUT_DIR" \
  --output-dir "$POSITIONED_DIR" \
  --width "$CANVAS_WIDTH" \
  --height "$CANVAS_HEIGHT" \
  --scale "$OBJECT_SCALE" \
  --bottom-margin "$BOTTOM_MARGIN"

source openai_meta/bin/activate



python inference_openai.py \
  --output_data_path "$Prompt_Model_Output_Path" \
  --generate_nums "$prompt_nums" \
  --image_dir "$POSITIONED_DIR" \
  --gpt-model "gpt-5.4-mini" \
  --product-name "$PRODUCT_NAME" \
  --product-description "$PRODUCT_DESCRIPTION" \
  --seller-description "$SELLER_DESCRIPTION" \
  --store-info "$STORE_INFO" \
  --additional-request "$ADDITIONAL_REQUEST" \
  # --temperature 0.0

deactivate

# 4. Run CAIG image generation
source caig/bin/activate

# dir="$OUTPUT_PATH/construction_new_prompt/0713/add_syprompt"
dir="$OUTPUT_PATH/bakery/0713"
mkdir -p "$dir"

NEGATIVE_PROMPT="people, hands, face, body parts, duplicate foreground subject, similar duplicate product, text, typography, letters, numbers, logo, label, sign, price tag, watermark, floating objects, cluttered foreground, intersecting edges, malformed perspective, inconsistent scale, mismatched lighting, harsh heavy shadow, cartoon, illustration, CGI, 3D render, low quality, blurry"

OUTPUT_NAME="no_neg"


accelerate launch sample_llava.py \
  --batch_size 1 \
  --base_model_path "$base_model_path" \
  --save_path "$dir" \
  --controlnet_model_path "$controlnet_model_path" \
  --num_inference_steps 100 \
  --negative_prompt "$NEGATIVE_PROMPT" \
  --sampler_name "Euler a" \
  --data_path "$Prompt_Model_Output_Path" \
  --width "$CANVAS_WIDTH" \
  --height "$CANVAS_HEIGHT" \
  --image_scale 1.0 \
  --keep_loc \
  --output_name "$OUTPUT_NAME"

# # 5. Evaluate generated images with CLIPScore
# EVAL_OUTPUT_DIR="$OUTPUT_PATH/eval_clip_score"
# mkdir -p "$EVAL_OUTPUT_DIR"

# python scripts/eval_clip_score.py \
#   --prompt-json "$Prompt_Model_Output_Path" \
#   --image-dir "$dir" \
#   --store-info "$STORE_INFO" \
#   --output-json "$EVAL_OUTPUT_DIR/clip_scores.json" \
#   --output-csv "$EVAL_OUTPUT_DIR/clip_scores.csv"

deactivate