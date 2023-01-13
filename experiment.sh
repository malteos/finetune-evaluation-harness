
#!/bin/bash
# Exit if any error
set -o errexit

if [ $# -ne 3 ] ; then
    echo "usage: $0 <TASK TYPE (classification, ner, qa)> <MODEL_NAME> <DATASET_NAME>"
    exit
fi

# path to save the checkpoints and other config files
base_checkpoint_dir="/netscratch/agautam/experiments/test_logs"

# path to save the json results file (different from previous one)
base_log_dir="/netscratch/agautam/experiments/logs/evaluation_freeze"

max_seq_length=512
epochs=1
task_name=$1
model_name=$2
dataset_name=$3
per_device_train_batch_size=4


if [ ${task_name} = "classification" ]
then
    filename="hgf_fine_tune_class.py"

elif [ ${task_name} = "ner" ]
then
    filename="hgf_fine_tune_ner.py"

elif [ ${task_name} = "qa" ]
then
    filename="hgf_fine_tune_qa.py"
fi

echo "${filename}"
file_identifier=$RANDOM
echo "$file_identifier"

log_dir="$base_log_dir"/${task_name}_$file_identifier
echo "${log_dir}"


python "$filename" \
    --model_name_or_path "$model_name" \
    --dataset_name "$dataset_name" \
    --output_dir "$base_checkpoint_dir" \
    --do_train \
    --do_eval \
    --overwrite_output_dir True \
    --num_train_epochs ${epochs} \
    --max_seq_length ${max_seq_length} \
    --results_log_path ${log_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --freeze_layers True \
