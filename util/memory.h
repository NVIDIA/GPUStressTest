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

/**
 * \file
 * \brief C++ interface to CUDA device memory management functions.
 */

#include <memory>

#include "exceptions.h"

namespace cublas {
namespace device_memory {

/******************************************************************************
 * Allocation lifetime
 ******************************************************************************/

/// Allocate a buffer of \p count elements of type \p T on the current CUDA device
template <typename T>
T* allocate(size_t count = 1) {
  T* ptr = 0;
  size_t bytes = sizeof(T) * count;

  cudaError_t cuda_error = cudaMalloc((void**)&ptr, bytes);
  if (cuda_error != cudaSuccess) {
    throw cuda_exception(cuda_error, "Failed to allocate memory");
  }

  return ptr;
}

/// Free the buffer pointed to by \p ptr
template <typename T>
void free(T* ptr) {
  if (ptr) {
    cudaError_t cuda_error = (cudaFree(ptr));
    if (cuda_error != cudaSuccess) {
      throw cuda_exception(cuda_error, "Failed to free device memory");
    }
  }
}

/******************************************************************************
 * Data movement
 ******************************************************************************/

template <typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
  size_t bytes = count * sizeof(T);
  if (bytes == 0 && count > 0)
    bytes = 1;
  cudaError_t cuda_error = (cudaMemcpy(dst, src, bytes, kind));
  if (cuda_error != cudaSuccess) {
    throw cuda_exception(cuda_error, "cudaMemcpy() failed");
  }
}

template <typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToHost);
}

template <typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToDevice);
}

template <typename T>
void copy_host_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToHost);
}

/// Copies elements from device memory to host-side range
template <typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
  size_t elements = end - begin;
  copy_to_host(&*begin, device_begin, elements);
}

/// Copies elements to device memory from host-side range
template <typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
  size_t elements = end - begin;
  copy_to_device(device_begin, &*begin, elements);
}
} // namespace device_memory 
} // namespace cublas
