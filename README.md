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
gpu T4 test FP16 size 2 A 2813718656 B 3412772992 C 1231479872 targetgb 16 ratio 0.868 good 1
gpu T4 test C32 size 8 A 697352000 B 1001864000 C 187759168 targetgb 16 ratio 0.879 good 1
gpu T4 test FP32 size 2 A 3096336000 B 3972192000 C 331046912 targetgb 16 ratio 0.861 good 1
gpu T4 test FP64 size 16 A 481661360 B 361246020 C 108288192 targetgb 16 ratio 0.886 good 1
gpu T4 test FP32 size 2 A 3781723140 B 3181913280 C 300617408 targetgb 16 ratio 0.846 good 1
gpu A100 test INT8 size 1 A 26187688266 B 11047658931 C 2782772334 targetgb 40 ratio 0.932 good 1
gpu A100 test FP16 size 2 A 9434328852 B 7759862096 C 2959271952 targetgb 40 ratio 0.938 good 1
gpu A100 test TF32 size 2 A 3627617792 B 1178813952 C 14427872064 targetgb 40 ratio 0.896 good 1
gpu A100 test FP64 size 16 A 1547399980 B 706355900 C 252219380 targetgb 40 ratio 0.934 good 1
gpu A100 test FP32 size 2 A 3401931264 B 883822080 C 13235751680 targetgb 40 ratio 0.816 good 1
gpu A100 test INT8 size 1 A 34878833064 B 16672535724 C 28662757344 targetgb 80 ratio 0.934 good 1
gpu A100 test FP16 size 2 A 13000629632 B 13114567936 C 13557084032 targetgb 80 ratio 0.924 good 1
gpu A100 test TF32 size 2 A 18964675584 B 6192059904 C 14359400064 targetgb 80 ratio 0.920 good 1
gpu A100 test FP64 size 16 A 2127239680 B 1494054400 C 1361222080 targetgb 80 ratio 0.928 good 1
gpu A100 test FP32 size 2 A 22580150528 B 9977244160 C 7782231680 targetgb 80 ratio 0.939 good 1
gpu K80 test FP16 size 2 A 3464434400 B 1685441300 C 256463200 targetgb 11 ratio 0.915 good 1
gpu K80 test C32 size 8 A 719518800 B 539639100 C 107640300 targetgb 11 ratio 0.926 good 1
gpu K80 test FP32 size 2 A 2717929400 B 2299265300 C 340418200 targetgb 11 ratio 0.907 good 1
gpu K80 test FP64 size 16 A 383839200 B 262215360 C 24511080 targetgb 11 ratio 0.908 good 1
gpu K80 test FP32 size 2 A 3127660800 B 2165025600 C 116625300 targetgb 11 ratio 0.916 good 1
gpu M60 test FP16 size 2 A 1846001664 B 1885261248 C 346963968 targetgb 8 ratio 0.950 good 1
gpu M60 test C32 size 8 A 499487160 B 407690620 C 78645792 targetgb 8 ratio 0.918 good 1
gpu M60 test FP32 size 2 A 1900663264 B 1899702848 C 250525568 targetgb 8 ratio 0.943 good 1
gpu M60 test FP64 size 16 A 277820928 B 208365696 C 15925248 targetgb 8 ratio 0.935 good 1
gpu M60 test FP32 size 2 A 2166998240 B 1590727680 C 260333568 targetgb 8 ratio 0.936 good 1
gpu P40 test FP16 size 2 A 5111465860 B 3826974600 C 1854281160 targetgb 22 ratio 0.914 good 1
gpu P40 test C32 size 8 A 1293393457 B 962910975 C 460957575 targetgb 22 ratio 0.920 good 1
gpu P40 test FP32 size 2 A 5024275828 B 3761695080 C 1854281160 targetgb 22 ratio 0.901 good 1
gpu P40 test FP64 size 16 A 723929349 B 542961559 C 116086971 targetgb 22 ratio 0.937 good 1
gpu P40 test FP32 size 2 A 6251931657 B 4348479975 C 303325575 targetgb 22 ratio 0.923 good 1
gpu P100 test FP16 size 2 A 2813718656 B 3412772992 C 1231479872 targetgb 16 ratio 0.868 good 1
gpu P100 test C32 size 8 A 697352000 B 1001864000 C 187759168 targetgb 16 ratio 0.879 good 1
gpu P100 test FP32 size 2 A 3096336000 B 3972192000 C 331046912 targetgb 16 ratio 0.861 good 1
gpu P100 test FP64 size 16 A 481661360 B 361246020 C 108288192 targetgb 16 ratio 0.886 good 1
gpu P100 test FP32 size 2 A 3781723140 B 3181913280 C 300617408 targetgb 16 ratio 0.846 good 1
gpu H100 test INT8 size 1 A 43425053064 B 16672535724 C 35685877344 targetgb 95 ratio 0.939 good 1
gpu H100 test FP16 size 2 A 16364949632 B 13114567936 C 17065404032 targetgb 95 ratio 0.913 good 1
gpu H100 test TF32 size 2 A 23486275584 B 6192059904 C 17783000064 targetgb 95 ratio 0.931 good 1
gpu H100 test FP64 size 16 A 2707079680 B 1494054400 C 1732262080 targetgb 95 ratio 0.931 good 1
gpu H100 test FP32 size 2 A 27684470528 B 9977244160 C 9541431680 targetgb 95 ratio 0.926 good 1
gpu V100 test FP16 size 2 A 2813718656 B 3412772992 C 1231479872 targetgb 16 ratio 0.868 good 1
gpu V100 test C32 size 8 A 697352000 B 1001864000 C 187759168 targetgb 16 ratio 0.879 good 1
gpu V100 test FP32 size 2 A 3096336000 B 3972192000 C 331046912 targetgb 16 ratio 0.861 good 1
gpu V100 test FP64 size 16 A 481661360 B 361246020 C 108288192 targetgb 16 ratio 0.886 good 1
gpu V100 test FP32 size 2 A 3781723140 B 3181913280 C 300617408 targetgb 16 ratio 0.846 good 1
gpu V100 test FP16 size 2 A 4075290624 B 10871635968 C 1146175488 targetgb 32 ratio 0.937 good 1
gpu V100 test C32 size 8 A 1120193760 B 2580355000 C 346458000 targetgb 32 ratio 0.942 good 1
gpu V100 test FP32 size 2 A 9419152000 B 5773564000 C 1023561472 targetgb 32 ratio 0.944 good 1
gpu V100 test FP64 size 16 A 522517440 B 515100000 C 912960000 targetgb 32 ratio 0.908 good 1
gpu V100 test FP32 size 2 A 7102594408 B 6086395056 C 853743072 targetgb 32 ratio 0.817 good 1
gpu Generic test FP16 size 2 A 1846001664 B 1885261248 C 346963968 targetgb 8 ratio 0.950 good 1
gpu Generic test C32 size 8 A 499487160 B 407690620 C 78645792 targetgb 8 ratio 0.918 good 1
gpu Generic test FP32 size 2 A 1900663264 B 1899702848 C 250525568 targetgb 8 ratio 0.943 good 1
gpu Generic test FP64 size 16 A 277820928 B 208365696 C 15925248 targetgb 8 ratio 0.935 good 1
gpu Generic test FP32 size 2 A 2166998240 B 1590727680 C 260333568 targetgb 8 ratio 0.936 good 1


