# Dev within Slurm compute job

See https://gist.github.com/malteos/5fe791fe10bb55028a02952d5f394bb3

## Build images

```bash
# from new dev image
srun \
  --container-image=$IMAGE \
  --container-save=$IMAGE --mem=150G \
  -p $ANY_PARTITION --pty /bin/bash

srun \
  --container-image=$IMAGE \
  --container-save=$DEV_IMAGE --mem=150G \
  -p $ANY_PARTITION --pty /bin/bash

srun \
  --container-image=$DEV_IMAGE \
  --container-save=$DEV_IMAGE --mem=150G \
  -p $ANY_PARTITION --pty /bin/bash

```

## Start job

```bash
export DEV_PARTITION=batch
export DEV_PARTITION=RTX6000,RTX3090

# with GPU
srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$DEV_IMAGE \
  --ntasks=1 --nodes=1 -p $DEV_PARTITION --gpus=1 --export ALL \
  --job-name devjob --no-container-remap-root \
  --time 12:00:00 /usr/sbin/sshd -D -e

# CPU only
srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$DEV_IMAGE \
  --ntasks=1 --nodes=1 -p $DEV_PARTITION --gpus=0 --export ALL \
  --job-name devjob --no-container-remap-root \
  --time 12:00:00 /usr/sbin/sshd -D -e

```

## Environment

```bash
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```