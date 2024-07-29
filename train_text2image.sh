#!/usr/bin/bash

module load cray-python

accelerate launch train_text_to_image.py --train_data_dir images/ --validation_prompts "ps ps ps" "ar ar ar" "ar gs ps" --output_dir models/t2i/ --lr_scheduler cosine --dataloader_num_workers 4 --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --use_ema --resume_from_checkpoint latest --num_train_epochs 150