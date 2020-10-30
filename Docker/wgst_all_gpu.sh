#! /bin/bash


loop=${1-1}

num_gpu=$(nvidia-smi -L | wc -l)

docker pull nvcr.io/nvidian/wgst:1.1


for gpu in $(seq 0 $(expr $num_gpu - 1))
do
	echo gpu $gpu dev $dev
        docker tag nvcr.io/nvidian/wgst:1.1 wgst$gpu
	docker run -i --gpus device=$gpu --device=/dev/nvidiactl --device=/dev/nvidia-uvm nvcr.io/nvidian/wgst:1.1 $loop &
done


docker ps | grep wgst
