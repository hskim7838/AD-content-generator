#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH
export PYTHONWARNINGS="ignore:.*copying from a non-meta parameter.*:UserWarning"
# ===== paths =====
INPUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images_original_test"
CUTOUT_DIR="/home/ai3/AD-content/CAIG/tiny_dataset/images_test"
base_data_path="tiny_dataset/tiny_auto.json"
OUTPUT_PATH="./output_demo"

# ===== CAIG settings =====
prompt_nums=1
controlnet_model_path="lllyasviel/control_v11p_sd15_canny"
base_model_path="digiplay/majicMIX_realistic_v7"
Prompt_Model_Path="/home/ai3/AD-content/CAIG/llava-v1.6-vicuna-7b-pretrain"
Prompt_Model_Output_Path="$OUTPUT_PATH/output_prompt.json"

# ===== OpenAI settings =====
OPENAI_ENV="secrets/openai.env"
STORE_INFO="Local cafe bakery, warm premium advertising mood"
mkdir -p "$CUTOUT_DIR"
mkdir -p "$OUTPUT_PATH"

# 1. Load OpenAI key
source "$OPENAI_ENV"

# 2. Remove background
source rembg/bin/activate
./caig/bin/python3 scripts/remove_bg_rembg.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$CUTOUT_DIR"
deactivate
# 3. Generate tiny.json with OpenAI API
rm -f "$base_data_path"

source openai_meta/bin/activate
python scripts/make_tiny_json_openai.py \
  --image-dir "$CUTOUT_DIR" \
  --out "$base_data_path" \
  --path-prefix "$CUTOUT_DIR" \
  --store-info "$STORE_INFO" \
  --model "gpt-5.5" 
deactivate

# 4. Run CAIG prompt model
source caig/bin/activate

rm -f "$Prompt_Model_Output_Path"

accelerate launch inference_llava.py \
  --model-path "$Prompt_Model_Path" \
  --output_data_path "$Prompt_Model_Output_Path" \
  --generate_nums "$prompt_nums" \
  --base_data_path "$base_data_path" \
  --temperature 1.0

# 5. Run CAIG image generation
dir="$OUTPUT_PATH/output_image"

accelerate launch sample_llava.py \
  --batch_size 1 \
  --base_model_path "$base_model_path" \
  --save_path "$dir" \
  --controlnet_model_path "$controlnet_model_path" \
  --num_inference_steps 30 \
  --sampler_name "Euler a" \
  --data_path "$Prompt_Model_Output_Path"


# 6. Evaluate generated images with CLIPScore
EVAL_OUTPUT_DIR="$OUTPUT_PATH/eval_clip_score"
mkdir -p "$EVAL_OUTPUT_DIR"

python scripts/eval_clip_score.py \
  --prompt-json "$Prompt_Model_Output_Path" \
  --image-dir "$dir" \
  --store-info "$STORE_INFO" \
  --output-json "$EVAL_OUTPUT_DIR/clip_scores.json" \
  --output-csv "$EVAL_OUTPUT_DIR/clip_scores.csv"

deactivate