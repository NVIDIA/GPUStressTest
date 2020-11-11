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
/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ exception semantics for CUDA error codes
 */

#include <cuda_runtime.h>
#include <iosfwd>
#include <stdexcept>


namespace cublas {

/// C++ exception wrapper for CUDA \p cudaError_t
class cuda_exception : public std::exception {
 public:
  /// Constructor
  cuda_exception(cudaError_t err = cudaErrorUnknown, const char* msg = "") : msg(msg), err(err) {}

  /// Returns the underlying CUDA \p cudaError_t
  cudaError_t cudaError() const { return err; }

 protected:
  /// Explanatory string
  const char* msg;

  /// Underlying CUDA \p cudaError_t
  cudaError_t err;
};

/// Writes a cudaError_t to an output stream
inline std::ostream& operator<<(std::ostream& out, cudaError_t result) {
  return out << cudaGetErrorString(result);
}

/// Writes a cuda_exception instance to an output stream
inline std::ostream& operator<<(std::ostream& out, cuda_exception const& e) {
  return out << e.what() << ": " << e.cudaError();
}

static inline void
cuda_check_error(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw cuda_exception(err, msg);
  }
}

static inline const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}

/// C++ exception wrapper for cuBLAS \p cudaError_t
class cublas_exception : public std::exception {
 public:
  /// Constructor
  cublas_exception(cublasStatus_t err = CUBLAS_STATUS_SUCCESS, const char* msg = "") : msg(msg), err(err) {}

  /// Returns the underlying CUDA \p cudaError_t
  cublasStatus_t cublasError() const { return err; }

 protected:
  /// Explanatory string
  const char* msg;

  /// Underlying cuBLAS \p cudaError_t
  cublasStatus_t err;
};

/// Writes a cublasStatus_t to an output stream
static inline std::ostream& operator<<(std::ostream& out, cublasStatus_t result) {
  return out << cublasGetErrorString(result);
}

static inline void
cublas_check_error(cublasStatus_t err, const char* msg) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    throw cublas_exception(err, msg);
  }
}


/// Writes a cublas_exception instance to an output stream
static inline std::ostream& operator<<(std::ostream& out, cublas_exception const& e) {
  return out << e.what() << ": " << e.cublasError();
}

}  // namespace cublas
