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
#include "test_util.h"

static __global__ void wait_kernel_impl(volatile int32_t* counter, int32_t threshold) {
  static const int64_t WAIT_CYCLES = 1024;
  static const int64_t CYCLES_PER_SECOND = 2000000000LL;
  int64_t tstart = clock64();
  while (*counter < threshold && (clock64() - tstart) < 10 * CYCLES_PER_SECOND) {
    int64_t elapsed = 0, t0 = clock64();
    do {
      elapsed = clock64() - t0;
    } while (elapsed < WAIT_CYCLES);
  }
}

void wait_kernel(volatile int32_t* counter, int32_t threshold) { wait_kernel_impl<<<1, 1>>>(counter, threshold); }

cudaError_t gpuAllocPinnedAndMap(size_t sizeInbytes, void** HostMemPtr,
                                        void** GpuMemPtr) {
  cudaError_t err = cudaHostAlloc(HostMemPtr, sizeInbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    return err;
  }
  return (cudaHostGetDevicePointer(GpuMemPtr, *HostMemPtr, 0));
}
