# Steps to set up the environment
1. Create conda environment with python version 3.9
   
   ```bash
   conda create -n myenv python=3.9
2. Activate envirnoment
   
   ```bash
   conda activate myenv
3. install requirements.txt

   ```bash
   pip install -r requirements.txt
   
4. install current version of opendelta from git (need to do this as version in original requirenemtn.txt file requires sklearn module which is deprecated)

   ```bash
   pip install git+https://github.com/thunlp/OpenDelta.git

5. Upgrade datasets transformers torch

   ```bash
    pip install --upgrade datasets==2.21.0 transformers torch

6. Install protobuf

   ```bash
   pip install protobuf==3.20.3

7. Test command to finetune deberta on rte on 4 samples

   ```bash
    python -u run_glue.py --do_eval --do_train --do_predict --task_name rte --max_train_samples 4 --max_eval_samples 5 --max_predict_samples 5 --eval_steps 1000 --evaluation_strategy steps --learning_rate 1.2e-3 --max_grad_norm 0.1  --logging_steps 100 --max_steps -1 --model_name_or_path microsoft/deberta-v3-base --num_train_epochs 17 --output_dir results/rte-lambda2_0_7e-4_lambda_0.001_epoch_1_seed_48_1 --overwrite_output_dir --per_device_eval_batch_size 1 --per_device_train_batch_size 1 --save_steps 1000 --save_strategy steps --save_total_limit 1 --warmup_ratio 0.06 --warmup_steps 0 --weight_decay 0.1 --disable_tqdm false  --sparse_lambda 0.001 --sparse_lambda_2 0 --seed 48 --lora_r 8 --max_seq_length 256 --max_lambda 7e-4 --lambda_schedule linear --lambda_num 1 --train_sparse 
