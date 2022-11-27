# Dev within Slurm compute job

See https://gist.github.com/malteos/5fe791fe10bb55028a02952d5f394bb3

## Start job

```bash
export DEV_PARTITION=batch
export DEV_PARTITION=RTX6000,RTX3090,GTX1080Ti,RTX280Ti

# with GPU
srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$IMAGE \
  --ntasks=1 --nodes=1 -p $DEV_PARTITION --gpus=1 --export ALL \
  --job-name devjob --no-container-remap-root \
  --time 12:00:00 /usr/sbin/sshd -D -e

# CPU only
srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$IMAGE \
  --ntasks=1 --nodes=1 -p $DEV_PARTITION --gpus=0 --export ALL \
  --job-name devjob --no-container-remap-root \
  --time 12:00:00 /usr/sbin/sshd -D -e

```