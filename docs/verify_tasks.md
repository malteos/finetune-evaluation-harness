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
# - true: acc = 90
${SRUN} python src/finetune_eval_harness/main.py --tasks pawsx_en --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# pawsx_de
# - expected full train: acc = 85
# - true: acc = 83
${SRUN} python src/finetune_eval_harness/main.py --tasks pawsx_de --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xstance_de
# - expected full train: macro f1 = 76
# - true: acc = 64
${SRUN} python src/finetune_eval_harness/main.py --tasks xstance_de --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xstance_fr
# - expected full train: macro f1 = ?
# - true: acc = 70
${SRUN} python src/finetune_eval_harness/main.py --tasks xstance_fr --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 


# xtreme_panx_ner_en
# - expected full train: f1 = 85
# - true: acc = 82
${SRUN} python src/finetune_eval_harness/main.py --tasks xtreme_panx_ner_en --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# xtreme_panx_ner_es
# - expected full train: f1 = 77
# - true: acc = 90
${SRUN} python src/finetune_eval_harness/main.py --tasks xtreme_panx_ner_es --model_name_or_path ${MODELS_DIR}/bert-base-multilingual-cased \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 


# multi_eurlex_en
# - expected full train:  = 43
# - true: acc = 13
${SRUN} python src/finetune_eval_harness/main.py --tasks multi_eurlex_en --model_name_or_path ${MODELS_DIR}/xlm-roberta-base \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 

# multi_eurlex_fr
# - expected full train:  = 30
# - true: acc = 12
${SRUN} python src/finetune_eval_harness/main.py --tasks multi_eurlex_fr --model_name_or_path ${MODELS_DIR}/xlm-roberta-base \
     --output_dir ${VERIFY_DIR} --results_dir ${VERIFY_DIR}/metrics --report_to none --per_device_train_batch_size 64 --max_seq_length 256 --max_train_samples 10_000 --overwrite_cache 


```