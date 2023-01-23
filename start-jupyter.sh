#!/bin/bash

docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE -v $(pwd):/nwsi166-merlin nvcr.io/nvidia/merlin/merlin-tensorflow:22.11 /bin/bash -c 'cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='''
