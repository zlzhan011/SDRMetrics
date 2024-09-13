#!/bin/bash
set -e

CONTAINER_ID="benjijang/deepdfa:latest"
PWD=/data/cs_lzhan011/vulnerability/DeepDFA_V2/DeepDFA
#sudo docker pull $CONTAINER_ID
#mkdir -p DDFA/storage LineVul/linevul/data
sudo docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --name deepdfa-cpu \
    -v "$PWD/DDFA/storage:/DeepDFA/DDFA/storage" -v "$PWD/LineVul/linevul/data:/DeepDFA/LineVul/linevul/data" \
    $CONTAINER_ID
