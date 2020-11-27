To build gst in Linux, edit the CMakeLists.txt and set the location of CUDA 
(default /usr/local/cuda) and the path to your local gst repository:
e.g.
```
set(CUDA_HOME "/usr/local/cuda" CACHE STRING "" FORCE)
set(GPUStressTest_HOME "/home/dfisk/GPUStressTest" CACHE STRING "" FORCE)
```
There is one command line argument to gst: -T=n,   where is the loop count.
(default is -T=10)  This determines how long each test runs, up to 600 seconds
at which point the hang detection is triggered and the test aborts.

Each invocation of gst selected the first visible GPU and runs a series of 5 tests
loop count times and states PASS/FAIL for each.

Under the Docker directory are helpers to create and deploy GST as a container. 
With the addition of OpenMPI /mpirun the container can be run on 1 to all GPUs
on or more hosts. Please see the scripts in the Docker directory for examples.

In order to build GST on Windows 10 (for Server 2019 and 2020)
Install the POSIX compatibility package provided by Microsoft:
https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019
https://github.com/microsoft/vcpkg.git

see these commands:
vcpkg bootstrap-vcpkg.bat
vcpkg integrate install
vcpkg integrate project
vcpkg list
vcpkg install pthreads:x64-windows

The project properties in VSC need to point to the installation location of GPUStreesTest to find it’s util folder.


