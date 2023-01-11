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
#include <curand_kernel.h>

#include "common_header.h"
#include "fill_matrix.h"
#include "test_util.h"
#include "type_convert.h"

namespace impl {

namespace patterns {
enum type { pseudorandom, normal };
}

template <typename T, patterns::type Pattern>
struct generator;

template <typename T, bool IsInt = custd::is_integer<T>::value>
struct generator_pseudorandom {
  // version for integral types (int, int8_t)
  struct State {};

  static const int MAX = 100;

  int seed;

  generator_pseudorandom(int seed) : seed(seed) {}

  __inline__ __device__ __host__ T generate(State& state, int64_t rows,
                                            int64_t cols, int64_t lda,
                                            int64_t row, int64_t col,
                                            int batch) {
    int changedseed = (seed == 0) ? (row + 1) * (col + 1) * int(lda)
                                  : (row + 1) * (col + 1) * (seed + batch);
    changedseed = ((row + col) % 2 == 0) ? changedseed : -changedseed;

    return cuGet<T>((int)((int(lda) * row + col + changedseed) % MAX),
                    (int)((int(lda) * row + col + changedseed) % MAX));
  }
};

template <typename U>
struct quant_params {
  static constexpr int period = 253;
  static constexpr double range = 256.0;
};

template <>
struct quant_params<__nv_fp8_e4m3> {
  static constexpr int period = 62;
  static constexpr double range = 64.0;
};

template <typename T>
struct generator_pseudorandom<T, false> {
  // version for floating point types (non-integer)
  struct State {};

  int seed;

  generator_pseudorandom(int seed) : seed(seed) {}

  __inline__ __device__ __host__ void init(State& state, int subsequence) {}

  __inline__ __device__ __host__ T generate(State& state, int64_t rows,
                                            int64_t cols, int64_t lda,
                                            int64_t row, int64_t col,
                                            int batch) {
    return cuGet<T>(
        ((double)(((int(lda) * int(row) + int(col) + (seed + batch)) %
                   quant_params<T>::period) +
                  1)) /
            quant_params<T>::range,
        ((double)((((int(cols) * int(row) + int(col)) + 123 + (seed + batch)) %
                   quant_params<T>::period) +
                  1)) /
            quant_params<T>::range);
  }
};

template <typename T>
struct generator<T, patterns::pseudorandom> : generator_pseudorandom<T> {
  __inline__ __device__ __host__ void init(
      typename generator_pseudorandom<T>::State& state, int subsequence) {}

  generator(int seed, double mean, double sd)
      : generator_pseudorandom<T>(seed) {}
};

template <typename T>
struct generator<T, patterns::normal> {
  struct State {
    curandStateMRG32k3a_t state;
  };

  int seed;
  double sd;
  double mean;

  generator(int seed, double mean, double sd)
      : seed(seed), sd(sd), mean(mean) {}

  __inline__ __device__ __host__ void init(State& state, int subsequence) {
    curand_init(seed, subsequence, 0, &state.state);
  }

  __inline__ __device__ __host__ T generate(State& state, int64_t rows,
                                            int64_t cols, int64_t lda,
                                            int64_t row, int64_t col,
                                            int batch) {
    double2 val2 = curand_normal2_double(&state.state);
    return cuGet<T>(cuFma(val2.x, sd, mean), cuFma(val2.y, sd, mean));
  }
};

// grid and block sizes are fixed to device size, each thread writes a single
// pixel and then jumps over to next strip, until whole buffer is covered
template <typename T, class Generator>
__global__ void fillMatrixDevice_kernel(const bufferBatchVariant<T> buf,
                                        const size_t size, const int64_t lda,
                                        const int64_t rows, const int64_t cols,
                                        const cublasFillMode_t fillMode,
                                        const cublasDiagType_t diagType,
                                        Generator generator, const bool fillNaN,
                                        int batchCount) {
  const int64_t block_offset = gridDim.x * blockDim.x;

  const int64_t row_incr = block_offset % lda;
  const int64_t col_incr = block_offset / lda;

  const int64_t cursor_start = blockIdx.x * blockDim.x + threadIdx.x;

  const T valnan = cuMakeNaN<T>();
  typename Generator::State state;
  generator.init(state, cursor_start);

  for (int batch = 0; batch < batchCount; batch++) {
    int64_t cursor = cursor_start;
    T* p = batch_ptr(buf, batch) + cursor;
    int64_t row = cursor % lda;
    int64_t col = cursor / lda;

    for (; cursor < size; cursor += block_offset, p += block_offset,
                          row += row_incr, col += col_incr) {
      if (row >= lda) {
        row -= lda;
        col++;
      }

      if ((row < rows) && (col < cols) &&
          ((fillMode == CUBLAS_FILL_MODE_FULL) ||  //
           ((fillMode == CUBLAS_FILL_MODE_LOWER) && (col <= row)) ||
           ((fillMode == CUBLAS_FILL_MODE_UPPER) && (col >= row)))) {
        // fill matrix
        if ((diagType == CUBLAS_DIAG_NON_UNIT) || (row != col)) {
          *p = generator.generate(state, rows, cols, lda, row, col, batch);
        } else {
          *p = cuGet<T>(1);
        }
      } else {
        // fill outside matrix
        if (fillNaN) {
          *p = valnan;
        }
      }
    }
  }
}

template <typename T>
cudaError_t fillMatrixDevice_helper<T>::run(
    const bufferBatchVariant<T>& deviceBuf, const size_t size,
    const int64_t lda, const int64_t rows, const int64_t cols,
    const cublasFillMode_t fillMode, const cublasDiagType_t diagType,
    const char pattern, const int seed, const double mean, const double sd,
    const bool fillNaN, const int batchCount) {
  int device = 0;
  cublas::cuda_check_error(cudaGetDevice(&device), "cudaGetDevice failed");

  cudaDeviceProp prop;
  cublas::cuda_check_error(cudaGetDeviceProperties(&prop, device),
                           "cudaGetDeviceProperties failed");

  // smallest grid/block size that utilize whole device well
  const int grid_size = prop.multiProcessorCount * 4;
  const int block_size = prop.warpSize * 4;

  switch (pattern) {
    case 'r':
      fillMatrixDevice_kernel<<<grid_size, block_size>>>(
          deviceBuf, size, lda, rows, cols, fillMode, diagType,
          generator<T, patterns::normal>(seed, mean, sd), fillNaN, batchCount);
      break;
    case 'P':
    default:
      fillMatrixDevice_kernel<<<grid_size, block_size>>>(
          deviceBuf, size, lda, rows, cols, fillMode, diagType,
          generator<T, patterns::pseudorandom>(seed, mean, sd), fillNaN,
          batchCount);
  };

  return cudaGetLastError();
}
}  // namespace impl

