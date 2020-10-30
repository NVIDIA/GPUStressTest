#! /bin/bash

loop=${1:-8}

echo loop $loop

 export LD_LIBRARY_PATH=/nvidia
 /nvidia/wgst  -T=$loop

 exit 0


