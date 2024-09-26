# Original Adalora paper script
python ../examples/text-classification/run_glue.py \
--model_name_or_path microsoft/deberta-v3-base \
--task_name rte \
--apply_adalora --apply_lora \
--lora_type svd --target_rank 1  --lora_r 2  \
--reg_orth_coef 0.3 \
--init_warmup 600 --final_warmup 1800 --mask_interval 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 320 \
--per_device_train_batch_size 32 --learning_rate 1.2e-3 \
--num_train_epochs 50 --warmup_steps 200 \
--cls_dropout 0.20 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 100 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--seed 6 \
--root_output_dir ./output/glue/rte \
--overwrite_output_dir 

## Alora - Llama experiment script

# Set root output directory
ROOT_OUTPUT_DIR="./output/glue/rte/llama"

# Create the necessary directories
mkdir -p "$ROOT_OUTPUT_DIR/model"
mkdir -p "$ROOT_OUTPUT_DIR/log"

python ../examples/text-classification/run_glue.py \
--model_name_or_path NousResearch/Llama-2-7b-hf \
--task_name rte \
--apply_adalora --apply_lora \
--lora_type svd --target_rank 8  --lora_r 16  \
--reg_orth_coef 0.3 \
--init_warmup 600 --final_warmup 1800 --mask_interval 1 \
--beta1 0.85 --beta2 0.85 \
--lora_module query,key,value,intermediate,layer.output,attention.output \
--lora_alpha 32 \
--do_train --do_eval --max_seq_length 2048 \
--per_device_train_batch_size 16 --learning_rate 1e-4 \
--num_train_epochs 10 --warmup_steps 200 \
--cls_dropout 0.20 --weight_decay 0.01 \
--evaluation_strategy steps --eval_steps 200 \
--save_strategy steps --save_steps 10000 \
--logging_steps 10 \
--seed 6 \
--root_output_dir ./output/glue/rte/llama \
--overwrite_output_dir 