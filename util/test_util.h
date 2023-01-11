/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 NVIDIA
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
/* definition of int8 complex */
typedef struct __align__(2) {
  int8_t x;
  int8_t y;
}
cuInt8Complex;

namespace custd {
template <typename T, typename... Candidates>
struct is_same_type;
template <typename T, typename U>
struct is_same_type<T, U> {
  static const bool value = false;
};
template <typename T>
struct is_same_type<T, T> {
  static const bool value = true;
};
template <typename T, typename C, typename... Candidates>
struct is_same_type<T, C, Candidates...> {
  static const bool value =
      is_same_type<T, C>::value || is_same_type<T, Candidates...>::value;
};

template <class T, T v>
struct integral_constant {
  static const T value = v;
};

template <bool B>
struct bool_constant : integral_constant<bool, B> {};

template <typename T>
struct is_integer : bool_constant<is_same_type<T, int>::value ||       //
                                  is_same_type<T, unsigned>::value ||  //
                                  is_same_type<T, uint64_t>::value ||  //
                                  is_same_type<T, int64_t>::value ||   //
                                  is_same_type<T, uint16_t>::value ||  //
                                  is_same_type<T, int16_t>::value ||   //
                                  is_same_type<T, uint8_t>::value ||   //
                                  is_same_type<T, int8_t>::value ||    //
                                  is_same_type<T, cuInt8Complex>::value> {};
}  // namespace custd

void wait_kernel(volatile int32_t* counter, int32_t threshold);

static inline void wait_kernel_set_semaphore(int32_t* host_ptr, int32_t value) {
  *reinterpret_cast<volatile int32_t*>(host_ptr) = value;
}

static cudaError_t gpuAllocPinnedAndMap(size_t sizeInbytes, void** HostMemPtr,
                                        void** GpuMemPtr) {
  cudaError_t err = cudaHostAlloc(HostMemPtr, sizeInbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    return err;
  }
  return (cudaHostGetDevicePointer(GpuMemPtr, *HostMemPtr, 0));
}

