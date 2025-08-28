CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
nohup accelerate launch \
  --num_processes=8 \
  --main_process_port=29502 \
  --config_file finetune/accelerate_config/zero1.yaml \
  MOSS_TTSD/finetune/finetune_packing.py \
  --model_path MOSS_TTSD/fnlp/MOSS-TTSD-v0.5 \
  --train_data_dir  \
  --eval_data_dir  \
  --output_dir  --lora \
  > output.log 2>&1 &


