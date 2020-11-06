To buildi wgst, edit the CMakeLists.txt and set the location of CUDA 
(default /usr/local/cuda) and the path to your local wgst repository:
e.g.
set(CUDA_HOME "/usr/local/cuda" CACHE STRING "" FORCE)
set(WGST_HOME "/home/dgx2/dfisk/wgst/build/wgst" CACHE STRING "" FORCE)

There is one command line argument to wgst: -T=n,   where is the loop count.
(default is -T=10)  This determmins how long each test runs, upto 600 seconds
at which point the hang detection is triggered and the test aborts.

Each invocation of wgst selectes the first visible GPU and runs a series of 5 tests
loop count times and states PASS/FAIL for each.

