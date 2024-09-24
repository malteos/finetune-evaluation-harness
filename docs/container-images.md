

--no-container-remap-root \

srun     --container-image=/netscratch/enroot/podman+enroot.sqsh     --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"     --container-workdir="`pwd`"     --pty bash


srun   --no-container-remap-root  --container-image=/netscratch/enroot/podman+enroot.sqsh     --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER,"`pwd`":"`pwd`"     --container-workdir="`pwd`"     --pty bash


 podman build  --isolation=chroot --userns=keep-id  -t temp -f Dockerfile-debug .

podman build  --isolation=chroot -t temp -f Dockerfile-debug .
 
podman build  --uidmap 0:12345:1000 --gidmap 0:12345:1000 -t temp -f Dockerfile-debug .


podman build -t temp -f Dockerfile-debug .


srun  -p V100-16GB --mem=80G \
  enroot import \
  -o /netscratch/$USER/enroot/malteos_finetune-evaluation-harness.sqsh \
  'docker://ghcr.io#malteos/finetune-evaluation-harness:latest'
