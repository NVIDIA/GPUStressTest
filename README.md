To build gst in Linux, edit the CMakeLists.txt and set the location of CUDA 
(default /usr/local/cuda) and the path to your local gst repository:
e.g.
```
set(CUDA_HOME "/usr/local/cuda" CACHE STRING "" FORCE)
set(GPUStressTest_HOME "/home/dfisk/GPUStressTest" CACHE STRING "" FORCE)
```
There is one command line argument to gst: ```-T=n```   where n is the loop count.
(default is -T=10)  This determines how long each test runs, up to 600 seconds
at which point the hang detection is triggered and the test aborts.

Each invocation of gst selectes the first visible GPU and runs a series of 5 tests
loop count times and states PASS/FAIL for each.

Under the Docker directory there are helpers to create and deploy gst as a container. 
With the addition of OpenMPI /mpirun the container can be run on 1 to all GPUs
on or more hosts. Please see the scripts in the Docker directory for examples.

In order to build gst.exe on Windows 10 (also for Server 2019 and 2020)
Install the POSIX compatibility package provided by Microsoft:
```
https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019
https://github.com/microsoft/vcpkg.git

see these commands:
vcpkg bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg integrate project
vcpkg list
vcpkg install pthreads:x64-windows
```

The project build properties in VSC need to be modified to point to the installation location of GPUStreesTest to find it’s util folder.

To allow all supported GPU types to mock execution to tune / check matrix sizes use:
$ cd build
$ cmake -DDEBUG_MATRIX_SIZES:BOOL="ON" ..
$ ./gst > gst.out
$ ../util/parse_memory_targets.bash

