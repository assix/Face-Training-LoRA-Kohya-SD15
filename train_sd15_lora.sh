#!/bin/bash

# This script trains a LoRA model for Stable Diffusion 1.5
# It assumes you are running it from the directory containing FriendFace_Lora
# and that your kohya_ss environment is set up separately.

# --- Configuration ---
# Adjust this path if your kohya_ss installation is elsewhere
# Assuming kohya_ss is located in the home directory for portability
KOHYA_SS_DIR="~/kohya_ss"
# The project directory is assumed to be where this script is run from (./)
PROJECT_DIR="$(pwd)"
# --- End Configuration ---

# Expand the tilde to the full path for cd
KOHYA_SS_DIR_EXPANDED=$(eval echo "$KOHYA_SS_DIR")

echo "Navigating to kohya_ss directory: $KOHYA_SS_DIR_EXPANDED"
cd "$KOHYA_SS_DIR_EXPANDED" || { echo "Error: kohya_ss directory not found at $KOHYA_SS_DIR_EXPANDED"; exit 1; }

echo "Activating virtual environment..."
source ./venv/bin/activate

echo "Starting LoRA training..."
# All training paths now refer to the current project structure,
# removing the need for /Github-Published/ and /TrainLora/
accelerate launch sd-scripts/train_network.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="$PROJECT_DIR/images_512" \
  --output_dir="$PROJECT_DIR/model_512" \
  --logging_dir="$PROJECT_DIR/log_512" \
  --resolution="512,512" \
  --optimizer_type="AdamW8bit" \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=1200 \
  --network_dim=128 \
  --network_alpha=64 \
  --save_every_n_epochs=1 \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --seed="1337" \
  --cache_latents \
  --gradient_checkpointing \
  --save_model_as="safetensors" \
  --network_module=networks.lora

echo "Training finished."