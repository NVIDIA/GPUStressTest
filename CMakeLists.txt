cmake_minimum_required(VERSION 3.8.0)

set(CUDA_HOME "/usr/local/cuda" CACHE STRING "" FORCE)
set(GPUStressTest_HOME "/home/josephw/GPUStressTest" CACHE STRING "" FORCE)
set(LD_LIBRARY_PATH "${CUDA_HOME}/lib64;${LD_LIBRARY_PATH}")
set(CMAKE_VERBOSE_MAKEFILE "ON" CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "" FORCE)


project(gst LANGUAGES CXX CUDA C)

# For debugging matrix sizes; cmake -DDEBUG_MATRIX_SIZES:BOOL="ON" ..
set(DEBUG_MATRIX_SIZES "OFF" CACHE STRING "")

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


#Config CUDA for this project:

set(FLAGS "")
set(CMAKE_CUDA_FLAGS "" CACHE STRING "" FORCE)
# Note that cmake will insert semicolons between these item automatically...
set(SM_LIST_IN "75,80,90" CACHE STRING "Please input compute capability")
MARK_AS_ADVANCED(SM_LIST)
string(REPLACE "\," "\ " DISPLAY_SMS_LIST ${SM_LIST_IN})
string(REPLACE "\," "\;" SM_LIST ${SM_LIST_IN})

message("SMS_LIST:" ${SMS_LIST})
foreach(SM ${SMS_LIST})
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${SM},code=sm_${SM}")
endforeach(SM)
message("-- GPU SM = " ${DISPLAY_SMS_LIST})
message("-- CMAKE_CUDA_FLAGS = " ${CMAKE_CUDA_FLAGS})

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(FLAGS ${FLAGS} "-G")
    set(FLAGS ${FLAGS} -Xcompiler -g)
endif()


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
MARK_AS_ADVANCED(CLEAR CMAKE_CUDA_FLAGS)

# Global CXX/CUDA flags
set(CUDALIB_CUDA_CXX_FLAGS
    "-Wall -Wextra -fno-strict-aliasing -Wno-unused-parameter -I${CUDA_HOME}/include -I${CUDA_HOME}/cublasLt -L${CUDA_HOME}/lib64"
    # -fvisibility=hidde}
    CACHE STRING "Common CXX/CUDA flags")

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDALIB_CUDA_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-DDEBUG=1")
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined -Wl,--no-allow-shlib-undefined")

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(CMAKE_CUDA_COMPILER_ID MATCHES CLANG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDALIB_CUDA_CXX_FLAGS} -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
else() #NVCC
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -Xcompiler \"${CUDALIB_CUDA_CXX_FLAGS}\"")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
endif()

if(DEBUG_MATRIX_SIZES MATCHES ON)
message("DEBUG_MATRIX_SIZES ")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDEBUG_MATRIX_SIZES")
endif()


get_filename_component(CUDA_COMPILER_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_COMPILER_DIR}/..)
message(status " CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/util)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${CUDA_HOME}/cublasLt)

LIST(APPEND SRC_FILES
    ${PROJECT_SOURCE_DIR}/util/test_args.cu
    ${PROJECT_SOURCE_DIR}/util/measure.cc
    ${PROJECT_SOURCE_DIR}/util/device_info.cu
    ${PROJECT_SOURCE_DIR}/util/test_util.cu
    )

add_executable(gst main.cu ${SRC_FILES})
set_target_properties(gst PROPERTIES CUDA_ARCHITECTURES ${SM_LIST})
target_link_libraries(gst cublas cublasLt)



