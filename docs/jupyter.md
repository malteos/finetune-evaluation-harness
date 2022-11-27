# Jupyter

```bash

srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME --container-workdir=${BASE_DIR} --container-image=$IMAGE \
  --ntasks=1 --nodes=1 -p RTX6000 --gpus=1 --export ALL --pty bash


echo "Jupyter starting at ... http://${HOSTNAME}.kl.dfki.de:8880" && jupyter notebook --ip=0.0.0.0 --port=8880 \
    --allow-root --no-browser --config /home/mostendorff/.jupyter/jupyter_notebook_config.json \
    --notebook-dir /netscratch/mostendorff/experiments


# start with fixed token (for VSCode -> "Specify Jupyter connection")
JUPYTER_TOKEN=opengptx jupyter notebook

```