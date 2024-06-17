export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
# export INSTANCE_DIR="../text_to_image/fill-50k-100"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3"

# SD3 dreambooth
# time accelerate launch train_dreambooth_sd3.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --mixed_precision="bf16" \
#   --instance_prompt="a photo in ukj style" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --learning_rate=1e-4 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --use_8bit_adam \
#   --max_train_steps=100 \
#   --seed="0"


# SD3 lora
time accelerate launch train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo in ukj style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --use_8bit_adam \
  --max_train_steps=100 \
  --rank 4 \
  --seed="0"
