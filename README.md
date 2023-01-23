# NSWI166-merlin

## Requirements

- `Docker` installed
- (optionally) Docker GPU support set up, see <https://docs.docker.com/config/containers/resource_constraints/#gpu>

## Instructions

Run `start-jupyter.sh` to start Jupyter lab server on localhost:

```bash
sudo ./start-jupyter.sh
```

or by calling the Docker manually:

```bash
sudo docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE -v $(pwd):/nwsi166-merlin nvcr.io/nvidia/merlin/merlin-tensorflow:22.11 /bin/bash -c 'cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='''
```

In case the `--gpus all` is not supported (e.g. because no NVIDIA GPU is present or the docker is not set up to expose it), the command in `start-jupyter.sh`
can be adapted:

```bash
sudo docker run --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE -v $(pwd):/nwsi166-merlin nvcr.io/nvidia/merlin/merlin-tensorflow:22.11 /bin/bash -c 'cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='''
```

If successful, the running instance should be found on <http://localhost:8888/lab> by default. There, you can find examples provided by NVIDIA as a part of the container image. Our experiment can be found in `/nswi166-merlin directory`: <http://127.0.0.1:8888/lab/tree/nwsi166-merlin>
