/******************************************************************************
 * Copyright (c) 1993-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are not permitted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
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
#include <cuda_runtime.h>
#include <stdint.h>

void wait_kernel(volatile int32_t* counter, int32_t threshold);

static inline void wait_kernel_set_semaphore(int32_t* host_ptr, int32_t value) {
  *reinterpret_cast<volatile int32_t*>(host_ptr) = value;
}

cudaError_t gpuAllocPinnedAndMap(size_t sizeInbytes, void** HostMemPtr,
                                        void** GpuMemPtr);

template <typename T>
static T roundoff(T x, unsigned int granul) {
  return granul * ((x + (granul - 1)) / granul);
}

template <int VSCALE_SIZE>
static void getSFDimensions(int rows, int cols, int& sf_rows, int& sf_cols) {
  const size_t SFX_BLOCK_COLS = 32;
  const size_t SFX_BLOCK_ROWS = 4;
  const size_t SFX_BLOCK_INNER = 4;
  const size_t BLOCK_ROWS = SFX_BLOCK_INNER * VSCALE_SIZE;
  const size_t BLOCK_COLS = SFX_BLOCK_COLS * SFX_BLOCK_ROWS;

  sf_rows = roundoff(rows, BLOCK_ROWS) / VSCALE_SIZE;
  sf_cols = roundoff(cols, BLOCK_COLS);
}
