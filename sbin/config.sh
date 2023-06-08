#!/usr/bin/env bash
#############################################
# Environment config
# Load env vars with `. sbin/config.sh`
#############################################

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Utils
#taill() { tail -f  $(ls -t | head -1); }
#wq() { watch -n 10 squeue -u $USER; }

#export IMAGE=/netscratch/mostendorff/enroot/leogao2+gpt-neox+sha-cbddb16.sqsh
#export IMAGE=/netscratch/mostendorff/enroot/malteos+oxw+latest.sqsh
#export IMAGE=/netscratch/mostendorff/enroot/malteos+oxw-bigs+latest.sqsh

export PROJECT_ID=finetune-evaluation-harness
#export BASE_TAG=22.08-py3
#export IMAGE=/netscratch/$USER/enroot/malteos+obmd+${BASE_TAG}.sqsh 
#export IMAGE=/netscratch/$USER/enroot/malteos+finetune-eval+latest.sqsh
export IMAGE=/netscratch/$USER/enroot/malteos_finetune-evaluation-harness.sqsh
export DEV_IMAGE=/netscratch/$USER/enroot/malteos_finetune-evaluation-harness_dev.sqsh

# Weights & Biases
export WANDB_PROJECT=finetune-eval # $PROJECT_ID

# Default
export SBATCH="sbatch"

# serv-9212 settings
export GPU2_REPOS_DIR=/data/experiments/mostendorff
export GPU2_BASE_DIR=${GPU2_REPOS_DIR}/$PROJECT_ID
export GPU2_DATASETS_DIR=/data/datasets

# Juwels settings
#export JUWELS_REPOS_DIR=/p/project/opengptx/ostendorff1
export JUWELS_REPOS_DIR=/p/project/opengptx-elm/ostendorff1
export JUWELS_BASE_DIR=${JUWELS_REPOS_DIR}/oxw
# syslinnk for data dir: project -> scratch
export JUWELS_DATASETS_DIR=/p/project/opengptx-elm/ostendorff1/datasets
#export JUWELS_RUN=/p/project/opengptx/ostendorff1/bigscience-code/run_scripts 
export JUWELS_RUN=$JUWELS_REPOS_DIR/juwels-setup/run_scripts
export JUWELS_RUN_STAGE2020=/p/project/opengptx/ostendorff1/bigscience-code/run_scripts 
alias juwels_sbatch="cd $JUWELS_RUN; sbatch --ntasks-per-node=1 --nodes=4 --gres=gpu:4 --cpus-per-task=48 --hint=nomultithread --time=0-12:00:00  --partition=booster --account=opengptx-elm"

# DFKI slurm specific settings
export SLURM_REPOS_DIR=/netscratch/mostendorff/experiments
export SLURM_BASE_DIR=${SLURM_REPOS_DIR}/$PROJECT_ID
export SLURM_DATASETS_DIR=/netscratch/mostendorff/datasets
export SLURM_MEM=150G
export SLURM_CPUS=8
export ANY_PARTITION=RTX6000,RTX3090,RTXA6000-SLT,A100-40GB,V100-32GB,V100-16GB
alias dfki_sbatch="sbatch --ntasks-per-node=1 --nodes=4 --gres=gpu:4 --cpus-per-task=20 --mem=150G --partition=A100 --time=0-12:00:00"


# TUD / Taurus settings
export TAURUS_REPOS_DIR=/beegfs/ws/1/maos247e-gptx
export TAURUS_BASE_DIR=${TAURUS_REPOS_DIR}/$PROJECT_ID
export TAURUS_DATASETS_DIR=${TAURUS_REPOS_DIR}/datasets
export TAURUS_RUN=/beegfs/ws/1/maos247e-gptx/BigScience-Setup_TUD/setup-opengptx

# GPU server
[[ -d ${GPU2_BASE_DIR} ]] && export REPOS_DIR=${GPU2_REPOS_DIR}
[[ -d ${GPU2_BASE_DIR} ]] && export BASE_DIR=${GPU2_BASE_DIR}
[[ -d ${GPU2_BASE_DIR} ]] && export DATASETS_DIR=${GPU2_DATASETS_DIR}
[[ -d ${GPU2_BASE_DIR} ]] && export PY=python
[[ -d ${GPU2_BASE_DIR} ]] && export WORKERS=50

# Juwels cluster
[[ -d ${JUWELS_BASE_DIR} ]] && export BASE_DIR=${JUWELS_BASE_DIR}
[[ -d ${JUWELS_BASE_DIR} ]] && export REPOS_DIR=${JUWELS_REPOS_DIR}
[[ -d ${JUWELS_BASE_DIR} ]] && export DATASETS_DIR=${JUWELS_DATASETS_DIR}
[[ -d ${JUWELS_BASE_DIR} ]] && alias xsbatch=juwels_sbatch

# Slurm cluster
[[ -d ${SLURM_BASE_DIR} ]] && export BASE_DIR=${SLURM_BASE_DIR}
[[ -d ${SLURM_BASE_DIR} ]] && export REPOS_DIR=${SLURM_REPOS_DIR}
[[ -d ${SLURM_BASE_DIR} ]] && export DATASETS_DIR=${SLURM_DATASETS_DIR}
[[ -d ${SLURM_BASE_DIR} ]] && alias xsbatch=dfki_sbatch

# Taurus cluster
[[ -d ${TAURUS_BASE_DIR} ]] && export BASE_DIR=${TAURUS_BASE_DIR}
[[ -d ${TAURUS_BASE_DIR} ]] && export REPOS_DIR=${TAURUS_REPOS_DIR}
[[ -d ${TAURUS_BASE_DIR} ]] && export DATASETS_DIR=${TAURUS_DATASETS_DIR}

# default PY
# <strike>1xA100 (40GB)</strike>
# RTX6000 (24GB)
export SRUN="srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$IMAGE \
  --ntasks=1 --nodes=1 -p RTX6000 --gpus=1 --cpus-per-gpu=${SLURM_CPUS} --mem=${SLURM_MEM} --export ALL "

export SRUN_CPU="srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$IMAGE \
  --ntasks=1 --nodes=1 -p ${ANY_PARTITION} --mem=${SLURM_MEM} --cpus-per-task=20 --export ALL "

[[ -d ${SLURM_BASE_DIR} ]] && export PY="${SRUN} python "
[[ -d ${SLURM_BASE_DIR} ]] && export WORKERS=0

# Slurm bash
export SLURM_BASH="${SRUN} --pty bash"
export SLURM_CPU_BASH="${SRUN_CPU} --pty bash"
export SLURM_TB="${SRUN_CPU} sbin/tensorboard.sh"
export SLURM_NB="${SRUN_CPU} sbin/jupyter.sh"

#################

#if [[ -z "$REPOS_DIR" ]] || [[ -z "$BASE_DIR" ]] || [[ -z "$PY" ]]; then
#    echo "Environment could not be loaded - no matching base dir was found." 1>&2
#    exit 1
#fi

# Server dependent settings
export VENV_DIR="${BASE_DIR}"/venv
#export MEGATRON_DEEPSPEED_REPO=${REPOS_DIR}/bigscience-Megatron-DeepSpeed-oxw  # sbatch override this with variables.bash
export MEGATRON_DEEPSPEED_REPO=${REPOS_DIR}/obmd  # sbatch override this with variables.bash

export BIGSCIENCE_REPO=${REPOS_DIR}/bigscience
export BIGS_WORKING_DIR=${BASE_DIR}/data/bigs

export TRANSFORMERS_CACHE="${DATASETS_DIR}/transformers_cache"
export HF_DATASETS_CACHE="${DATASETS_DIR}/hf_datasets_cache"
export FLAIR_CACHE_ROOT="${DATASETS_DIR}/flair_cache"
export TENSORBOARD_DIR=${BASE_DIR}/data/tensorboard
export MODELS_DIR="${DATASETS_DIR}/huggingface_transformers/pytorch"

export VERIFY_DIR=${BASE_DIR}/output/verify


# Path to neox fork
export NEOX_DIR=${REPOS_DIR}/gpt-neox-oxw
export LM_EVAL_HARNESS_DIR=${REPOS_DIR}/lm-eval-harness


# Print current config

echo "-------------"
echo "Config loaded... "
echo "BASE_DIR = ${BASE_DIR}"
echo "-------------"
