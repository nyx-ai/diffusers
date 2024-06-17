MODEL_NAME="runwayml/stable-diffusion-v1-5"
# MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
DATASET_PATH="fill50k-100"
OUT_DIR='out/out7'
STEPS=100

# SD15 LoRA
# time python train_text_to_image_lora.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --train_data_dir=$DATASET_PATH \
#     --resolution 512 \
#     --max_train_steps $STEPS \
#     --learning_rate=1e-04 \
#     --max_grad_norm=1 \
#     --train_batch_size 1 \
#     --lr_scheduler="constant" \
#     --center_crop \
#     --mixed_precision bf16 \
#     --checkpointing_steps $STEPS \
#     --use_8bit_adam \
#     --rank 256 \
#     --output_dir $OUT_DIR

# SDXL LoRA
# time python train_text_to_image_lora_sdxl.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --train_data_dir=$DATASET_PATH \
#     --resolution 1024 \
#     --max_train_steps $STEPS \
#     --learning_rate=1e-04 \
#     --max_grad_norm=1 \
#     --train_batch_size 1 \
#     --lr_scheduler="constant" \
#     --center_crop \
#     --mixed_precision bf16 \
#     --checkpointing_steps $STEPS \
#     --use_8bit_adam \
#     --rank 64 \
#     --output_dir $OUT_DIR


# SDXL Dreambooth
# time python train_text_to_image_sdxl.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --train_data_dir=$DATASET_PATH \
#     --resolution 512 \
#     --max_train_steps $STEPS \
#     --learning_rate=1e-04 \
#     --max_grad_norm=1 \
#     --train_batch_size 1 \
#     --lr_scheduler="constant" \
#     --center_crop \
#     --mixed_precision bf16 \
#     --checkpointing_steps $STEPS \
#     --use_8bit_adam \
#     --gradient_checkpointing \
#     --output_dir $OUT_DIR

# SD15 Dreambooth
# time python train_text_to_image.py \
#     --pretrained_model_name_or_path=$MODEL_NAME \
#     --train_data_dir=$DATASET_PATH \
#     --resolution 512 \
#     --max_train_steps $STEPS \
#     --learning_rate=1e-04 \
#     --max_grad_norm=1 \
#     --train_batch_size 1 \
#     --lr_scheduler="constant" \
#     --center_crop \
#     --mixed_precision bf16 \
#     --checkpointing_steps $STEPS \
#     --use_8bit_adam \
#     --output_dir $OUT_DIR

# SD15 Dreambooth Flax
time python train_text_to_image_flax.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_PATH \
    --revision flax \
    --resolution 512 \
    --max_train_steps $STEPS \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --train_batch_size 1 \
    --mixed_precision bf16 \
    --lr_scheduler="constant" \
    --center_crop \
    --output_dir $OUT_DIR
