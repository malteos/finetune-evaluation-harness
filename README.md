# finetune-evaluation-harness

TODO

## Docker

To run all scripts within our Slurm cluster, everything needs to be containerized with Docker.

Image:
- Name: `malteos/finetune-eval` (available at: https://hub.docker.com/repository/docker/malteos/finetune-eval ) 

CI/CD:
- We use GitHub actions to automatically build and push new Docker images (see `.github` directory)
- To trigger a new build just include the string `docker build` in your commit message. Example commit message `git commit -am "updated dependencies (docker build)"`
- Building a new image is  **only required when installing/changing the Python packages!** 
    For code changes a `git pull` is so sufficient since this repo is mounted into the container.


## Slurm

Import latest Docker image as enroot:
```bash
srun enroot import -o /netscratch/$USER/enroot/malteos+finetune-eval+latest.sqsh docker://malteos/finetune-eval:latest
```