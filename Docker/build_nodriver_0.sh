cp ../build/wgst software/
docker build -f Dockerfile_nodriver_0 -t nvcr.io/nvidian/wgst:1.1 .
echo example: last argument is loop count passed to entery point:
echo docker run -i --gpus device=0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm nvcr.io/nvidian/wgst:1.1 10

