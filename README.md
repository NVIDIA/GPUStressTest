# GST - GPU Stress Test

# Build GST

## Prerequisites

### Ubuntu/WSL

- install CUDA
- sudo apt install build-essential
- sudo apt install cmake


### Windows MSVC Version

- install CUDA 12.X
- install visual studio 2022
- install vcpkg and pthread (See below)
- add nvcc (Add Right Click on Project -> Build Dependecies -> Build Customizations. following box will pop-up. Mark check for Cuda 12.6 (current version) and Click OK, if you don't see it, add it.  it is in $(CUDA_PTH)\extras\visual_studio_integration\MSBuildExtensions\}
- Add Definition DEBUG_MATRIX_SIZES to project property

### Windows GNU Version (Working...)

- install MinGW https://phoenixnap.com/kb/install-gcc-windows
- install visual studio
- install cmake https://cmake.org/download/



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

## Compile

To allow all supported GPU types to mock execution to tune / check matrix sizes use:
```
$ mkdir build
$ cd build
$ cmake -DDEBUG_MATRIX_SIZES:BOOL="ON" ..
$ make

$ ./gst > gst.out
$ ../util/parse_memory_targets.bash 
gpu T4 test FP16 size 2 A 2454022656 B 3329856000 C 1982169600 targetgb 16 ratio 0.904 good 1
gpu T4 test C32 size 4 A 62952000 B 3918640000 C 6629568 targetgb 16 ratio 0.929 good 1
gpu T4 test FP32 size 4 A 3096336000 B 786420000 C 6554112 targetgb 16 ratio 0.906 good 1
gpu T4 test FP64 size 8 A 361406360 B 1608210200 C 36172192 targetgb 16 ratio 0.934 good 1
gpu T4 test FP32 size 4 A 3096336000 B 786420000 C 6554112 targetgb 16 ratio 0.906 good 1
gpu A100 test INT8 size 1 A 26187688266 B 13745489310 C 346232334 targetgb 40 ratio 0.938 good 1
gpu A100 test FP16 size 2 A 9434328852 B 9965640960 C 380045952 targetgb 40 ratio 0.921 good 1
gpu A100 test TF32 size 4 A 3627617792 B 2835819520 C 3470848064 targetgb 40 ratio 0.925 good 1
gpu A100 test FP64 size 8 A 1547399980 B 2982099000 C 106482180 targetgb 40 ratio 0.864 good 1
gpu A100 test FP32 size 4 A 3627617792 B 2835819520 C 3470848064 targetgb 40 ratio 0.925 good 1
gpu A100 test INT8 size 1 A 60576166266 B 5381609310 C 11856002334 targetgb 80 ratio 0.906 good 1
gpu A100 test FP16 size 2 A 241903232 B 6432943360 C 33853364432 targetgb 80 ratio 0.944 good 1
gpu A100 test TF32 size 4 A 5257027584 B 11572439040 C 2947416064 targetgb 80 ratio 0.921 good 1
gpu A100 test FP64 size 8 A 362979840 B 9673664000 C 150390240 targetgb 80 ratio 0.949 good 1
gpu A100 test FP32 size 4 A 5257027584 B 11572439040 C 2947416064 targetgb 80 ratio 0.921 good 1
gpu H100 test INT8 size 1 A 60576166266 B 5381609310 C 11856002334 targetgb 80 ratio 0.906 good 1
gpu H100 test FP16 size 2 A 241903232 B 6432943360 C 33853364432 targetgb 80 ratio 0.944 good 1
gpu H100 test TF32 size 4 A 5257027584 B 11572439040 C 2947416064 targetgb 80 ratio 0.921 good 1
gpu H100 test FP64 size 8 A 362979840 B 9673664000 C 150390240 targetgb 80 ratio 0.949 good 1
gpu H100 test FP32 size 4 A 5257027584 B 11572439040 C 2947416064 targetgb 80 ratio 0.921 good 1
gpu H200 test FP16 size 2 A 93955456 B 71215409920 C 85809472 targetgb 140 ratio 0.950 good 1
gpu H200 test C32 size 4 A 575352000 B 34418640000 C 532191168 targetgb 140 ratio 0.945 good 1
gpu H200 test FP32 size 4 A 575352000 B 34418640000 C 532191168 targetgb 140 ratio 0.945 good 1
gpu H200 test FP64 size 8 A 481661360 B 16038810200 C 480784192 targetgb 140 ratio 0.905 good 1
gpu H200 test FP8 size 1 A 3781723140 B 131854132800 C 1245717408 targetgb 140 ratio 0.911 good 1
gpu V100 test FP16 size 2 A 2454022656 B 3329856000 C 1982169600 targetgb 16 ratio 0.904 good 1
gpu V100 test C32 size 4 A 62952000 B 3918640000 C 6629568 targetgb 16 ratio 0.929 good 1
gpu V100 test FP32 size 4 A 3096336000 B 786420000 C 6554112 targetgb 16 ratio 0.906 good 1
gpu V100 test FP64 size 8 A 361406360 B 1608210200 C 36172192 targetgb 16 ratio 0.934 good 1
gpu V100 test FP32 size 4 A 3096336000 B 786420000 C 6554112 targetgb 16 ratio 0.906 good 1
gpu V100 test FP16 size 2 A 104595456 B 15382609920 C 16649472 targetgb 32 ratio 0.902 good 1
gpu V100 test C32 size 4 A 1120193760 B 6622150000 C 88914000 targetgb 32 ratio 0.912 good 1
gpu V100 test FP32 size 4 A 1120193760 B 6622150000 C 88914000 targetgb 32 ratio 0.912 good 1
gpu V100 test FP64 size 8 A 694217440 B 2575500000 C 606480000 targetgb 32 ratio 0.902 good 1
gpu V100 test FP32 size 4 A 1120193760 B 6622150000 C 88914000 targetgb 32 ratio 0.912 good 1
gpu Generic test FP16 size 2 A 2046305664 B 1826772480 C 37267968 targetgb 8 ratio 0.910 good 1
gpu Generic test C32 size 4 A 143292160 B 1787081200 C 9889792 targetgb 8 ratio 0.904 good 1
gpu Generic test FP32 size 4 A 3201664 B 2025052480 C 64768 targetgb 8 ratio 0.945 good 1
gpu Generic test FP64 size 8 A 211500828 B 757254960 C 4406048 targetgb 8 ratio 0.906 good 1
gpu Generic test FP32 size 4 A 3201664 B 2025052480 C 64768 targetgb 8 ratio 0.945 good 1
gpu H200 test FP16 size 2 A 93955456 B 71215409920 C 85809472 targetgb 140 ratio 0.950 good 1
gpu H200 test C32 size 4 A 575352000 B 34418640000 C 532191168 targetgb 140 ratio 0.945 good 1
gpu H200 test FP32 size 4 A 575352000 B 34418640000 C 532191168 targetgb 140 ratio 0.945 good 1
gpu H200 test FP64 size 8 A 481661360 B 16038810200 C 480784192 targetgb 140 ratio 0.905 good 1
gpu H200 test FP8 size 1 A 3781723140 B 131854132800 C 1245717408 targetgb 140 ratio 0.911 good 1
```

## Run gst on Windows

### Prerequisites

- cuda installed
- msvc redistribution installed (maybe no need)
- gst.exe + pthreadVC3.dll

