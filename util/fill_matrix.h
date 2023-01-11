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
#include "common_header.h"

template <typename T>
struct bufferBatchVariant {
  typedef T *pointer_type;
  typedef T *const *pointer_array_type;
  union {
    pointer_type strided;
    pointer_array_type ptrArray;
  } ptr;
  int64_t stride;
  enum { STRIDED, PTR_ARRAY } mode;

  bufferBatchVariant() : ptr(NULL), stride(0), mode(STRIDED) {}
  bufferBatchVariant(pointer_type ptr, int64_t stride)
      : stride(stride), mode(STRIDED) {
    this->ptr.strided = ptr;
  }
  bufferBatchVariant(pointer_array_type ptrArray) : stride(0), mode(PTR_ARRAY) {
    this->ptr.ptrArray = ptrArray;
  }
};

template <typename T>
static inline __host__ __device__ typename bufferBatchVariant<T>::pointer_type
batch_ptr(const bufferBatchVariant<T> &tensor, size_t batch_idx) {
  switch (tensor.mode) {
    case bufferBatchVariant<T>::STRIDED:
      return tensor.ptr.strided + batch_idx * tensor.stride;
    case bufferBatchVariant<T>::PTR_ARRAY:
      return tensor.ptr.ptrArray[batch_idx];
  }

  assert(0);
  return NULL;
}

template <typename T>
static inline bufferBatchVariant<T> make_bufferBatchVariant(T *strided_ptr,
                                                            int64_t stride) {
  return bufferBatchVariant<T>(strided_ptr, stride);
}

template <typename T>
static inline bufferBatchVariant<T> make_bufferBatchVariant(
    T *const *ptrArray) {
  return bufferBatchVariant<T>(ptrArray);
}

namespace impl {
// internal struct to avoid specifying all parameters in the various
// instantiations of the function template
template <typename T>
struct fillMatrixDevice_helper {
  static cudaError_t run(const bufferBatchVariant<T> &deviceBuf,
                         const size_t size, const int64_t lda,
                         const int64_t rows, const int64_t cols,
                         const cublasFillMode_t fillMode,
                         const cublasDiagType_t diagType, const char pattern,
                         const int seed, const double mean, const double sd,
                         const bool fillNaN, const int batchCount);
};
}  // namespace impl

template <typename T>
cudaError_t fillMatrixDevice(const bufferBatchVariant<T> &deviceBuf,
                             const size_t size, const int64_t lda,
                             const int64_t rows, const int64_t cols,
                             const cublasFillMode_t fillMode,
                             const cublasDiagType_t diagType,
                             const char pattern, const int seed,
                             const double mean, const double sd,
                             const bool fillNaN, const int batchCount) {
  return impl::fillMatrixDevice_helper<T>::run(
      deviceBuf, size, lda, rows, cols, fillMode, diagType, pattern, seed, mean,
      sd, fillNaN, batchCount);
}

