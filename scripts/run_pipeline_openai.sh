#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTHONWARNINGS="ignore:.*copying from a non-meta parameter.*:UserWarning"

# ===== paths =====
INPUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images"
CUTOUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images_cutout_sample"
OUTPUT_PATH="./output_demo"

Prompt_Model_Output_Path="$OUTPUT_PATH/output_prompt.json"

# ===== CAIG settings =====
prompt_nums=1
controlnet_model_path="lllyasviel/control_v11p_sd15_canny"
base_model_path="digiplay/majicMIX_realistic_v7"

# ===== OpenAI settings =====
OPENAI_ENV="secrets/openai.env"
# STORE_INFO="Local cafe bakery, warm premium advertising mood"

PRODUCT_RATIO=80
BRAND_RATIO=10
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
rm -f "$Prompt_Model_Output_Path"

source openai_meta/bin/activate

python inference_openai.py \
  --output_data_path "$Prompt_Model_Output_Path" \
  --generate_nums "$prompt_nums" \
  --image_dir "$CUTOUT_DIR" \
  --gpt-model "gpt-5.4-mini" \
  --product-ratio "$PRODUCT_RATIO" \
  --brand-ratio "$BRAND_RATIO" \
  --sales-ratio "$SALES_RATIO"

deactivate

# 4. Run CAIG image generation
source caig/bin/activate

dir="$OUTPUT_PATH/new_dataset"
mkdir -p "$dir"

accelerate launch sample_llava.py \
  --batch_size 1 \
  --base_model_path "$base_model_path" \
  --save_path "$dir" \
  --controlnet_model_path "$controlnet_model_path" \
  --num_inference_steps 100 \
  --sampler_name "Euler a" \
  --data_path "$Prompt_Model_Output_Path"

# 5. Evaluate generated images with CLIPScore
EVAL_OUTPUT_DIR="$OUTPUT_PATH/eval_clip_score"
mkdir -p "$EVAL_OUTPUT_DIR"

python scripts/eval_clip_score.py \
  --prompt-json "$Prompt_Model_Output_Path" \
  --image-dir "$dir" \
  --store-info "$STORE_INFO" \
  --output-json "$EVAL_OUTPUT_DIR/clip_scores.json" \
  --output-csv "$EVAL_OUTPUT_DIR/clip_scores.csv"

deactivate