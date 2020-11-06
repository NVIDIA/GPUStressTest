#! /bin/bash

loop=${1-1}

num_gpu=$(nvidia-smi -L | wc -l)
max_rank=$(expr $num_gpu - 1) 

if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK} ] 
then
	echo OMPI_COMM_WORLD_LOCAL_RANK is not set on $(hostname). Aborting.
	exit -1
fi
if [ ${OMPI_COMM_WORLD_LOCAL_RANK} -gt $max_rank ]
then
	echo OMPI_COMM_WORLD_LOCAL_RANK ${OMPI_COMM_WORLD_LOCAL_RANK} is greater than max_rank $max_rank on $(hostname). Aborting.
	exit -1
fi
gpu=${OMPI_COMM_WORLD_LOCAL_RANK}

docker pull nvcr.io/nvidian/wgst:1.1

echo gpu $gpu dev $dev
docker tag nvcr.io/nvidian/wgst:1.1 wgst$gpu
docker run -i --gpus device=$gpu --device=/dev/nvidiactl --device=/dev/nvidia-uvm nvcr.io/nvidian/wgst:1.1 $loop 
