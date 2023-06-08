# Evaluate tasks and compare the results to the literature

## Run

Run on 1x 24GB GPU.

```bash
# settings

# xnli_en
# - expected full train: acc = 80.8
# - true: acc = 69
${SRUN} python src/finetune_eval_harness/main.py --tasks xnli_en --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xnli_de
# - expected full train: acc = 70
# - true: acc = 63
${SRUN} python src/finetune_eval_harness/main.py --tasks xnli_de --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 


# pawsx_en
# - expected full train: acc = 94
# - true: acc = 
${SRUN} python src/finetune_eval_harness/main.py --tasks pawsx_en --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# pawsx_de
# - expected full train: acc = 85
# - true: acc = 
${SRUN} python src/finetune_eval_harness/main.py --tasks pawsx_de --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xstance_de
# - expected full train: macro f1 = 76
# - true: acc = 
${SRUN} python src/finetune_eval_harness/main.py --tasks xstance_de --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xstance_it
# - expected full train: macro f1 = 70
# - true: acc = 
${SRUN} python src/finetune_eval_harness/main.py --tasks xstance_it --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

```