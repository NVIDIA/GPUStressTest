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
#include "common_header.h"
#include "exceptions.h"

void get_device_version(int &device_version) {
  int device;
  struct cudaDeviceProp properties;

  cublas::cuda_check_error(cudaGetDevice(&device), "get device failed");
  cublas::cuda_check_error(cudaGetDeviceProperties(&properties, device), "get device properties failed");
  device_version = properties.major * 100 + properties.minor * 10;
}

int showDevices(int currentDevice) {
  int totalDevices;

  cublas::cuda_check_error(cudaGetDeviceCount(&totalDevices), "get device count failed");
  printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
  for (int i = 0; i < totalDevices; i++) {
    struct cudaDeviceProp prop;
    cublas::cuda_check_error(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties failed");
    printf(
        "device %d %s: sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, "
        "boardGroupID=%d, memBusWidth=%d, memBW %f GB/s, Product=%s\n",
        i,
        (i == currentDevice) ? "(current) " : "",
        prop.multiProcessorCount,
        prop.major,
        prop.minor,
        (float)prop.clockRate * 1e-3,
        (int)(prop.totalGlobalMem / (1024 * 1024)),
        (float)prop.memoryClockRate * 1e-3,
        prop.ECCEnabled,
        prop.multiGpuBoardGroupID,
        prop.memoryBusWidth,
        float(prop.memoryClockRate) * 2 * prop.memoryBusWidth / 8 * 1e3f / (1ll << 30),
        prop.name);
  }
  return 0;
}

size_t getDeviceMemory() {
  size_t dFreeMem, dTotalMem;
  cublas::cuda_check_error(cudaMemGetInfo(&dFreeMem, &dTotalMem), "cudaMemGetInfo failed");
  return dFreeMem;
}

bool is_fit_tune_hsh_for_hopper(const BlasOpts &blas_opts) {
  // 1. problem size and type matches "-P=hsh -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0" (P,m,n,k; others can be ignored)
  if ((blas_opts.input_type_a != CUDA_R_16F) ||
      (blas_opts.input_type_b != CUDA_R_16F) ||
      (blas_opts.input_type_c != CUDA_R_16F) ||
      (blas_opts.input_type_c != CUDA_R_16F) ||
      (blas_opts.output_type != CUDA_R_16F)  ||
      (blas_opts.math_type != CUDA_R_32F)) {
    return false;
  }
  if ((blas_opts.m != 12288) ||
      (blas_opts.n != 9216)  ||
      (blas_opts.k != 32768)) {
    return false;
  }

  // 2. device to match at least H20 and at most any sm90 GPU.
  int device_version = 0;
  get_device_version(device_version);
  if (device_version != 900) {
    return false;
  }

  // 3. cublas version (from cublasGetProperty) is 12.6.1
  int version = 0;
  cublas::cublas_check_error(cublasGetProperty(MAJOR_VERSION, &version), "Get MAJOR_VERSION failed!!!");
  if (version != 12) {
    return false;
  }
  cublas::cublas_check_error(cublasGetProperty(MINOR_VERSION, &version), "Get MINOR_VERSION failed!!!");
  if (version != 6) {
    return false;
  }
  cublas::cublas_check_error(cublasGetProperty(PATCH_LEVEL, &version), "Get PATCH_LEVEL failed!!!");
  if (version != 1) {
    return false;
  }

  return true;
}

void tune_hsh_algo_for_hopper(cublasLtHandle_t ltHandle, cublasLtMatmulAlgo_t &algo) {
  // -algo39 -m_tile24 -m_stages35 -m_swizzle1 -m_cga5
  const int32_t algoId = 39;
  cublas::cublas_check_error(cublasLtMatmulAlgoInit(ltHandle,  //
                                             CUBLAS_COMPUTE_32F,   // compute
                                             CUDA_R_32F,   // scale
                                             CUDA_R_16F,   // A
                                             CUDA_R_16F,   // B
                                             CUDA_R_16F,   // C
                                             CUDA_R_16F,   // D
                                             algoId,
                                             &algo), "cublasLtMatmulAlgoInit in tune_hsh_algo_for_hopper failed!!!");

  const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_256x128; // 24
  cublas::cublas_check_error(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                     &tileId, sizeof(tileId)), "Set CUBLASLT_ALGO_CONFIG_TILE_ID in tune_hsh_algo_for_hopper failed!!!");

  const cublasLtMatmulStages_t stage = CUBLASLT_MATMUL_STAGES_64xAUTO; // 35
  cublas::cublas_check_error(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                            &stage, sizeof(stage)), "Set CUBLASLT_ALGO_CONFIG_STAGES_ID in tune_hsh_algo_for_hopper failed!!!");

  const uint32_t swizzle = 1;
  cublas::cublas_check_error(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                          &swizzle, sizeof(swizzle)), "Set CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING in tune_hsh_algo_for_hopper failed!!!");

  const uint16_t cga = 5;
  cublas::cublas_check_error(cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                          &cga, sizeof(cga)), "Set CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID in tune_hsh_algo_for_hopper failed!!!");

}

#if (defined WIN32) || (defined _WIN32) || (defined _WINDOWS)
#include <windows.h>
long long getSystemMemory() {
  MEMORYSTATUSEX state;  // Requires >= win2k
  memset(&state, 0, sizeof(state));
  state.dwLength = sizeof(state);
  if (0 == GlobalMemoryStatusEx(&state)) {
    return 0;
  } else {
    return (long long)state.ullTotalPhys;
  }
}
#elif defined(__linux) || defined(__powerpc64__)
#include <sys/sysinfo.h>

long long getSystemMemory(void) {
  struct sysinfo s_info;
  sysinfo(&s_info);
  return (long long)s_info.totalram * (long long)s_info.mem_unit;
}
#elif defined(__APPLE__)
#include <sys/sysctl.h>

long long getSystemMemory(void) {
  int memmib[2] = {CTL_HW, HW_MEMSIZE};
  long long mem = (size_t)0;
  size_t memsz = sizeof(mem);

  /* NOTE: This may cap memory reported at 2GB */
  if (sysctl(memmib, 2, &mem, &memsz, NULL, 0) == -1) {
    return 0;
  } else {
    return mem;
  }
}
#elif defined(__QNX__)
long long getSystemMemory(void) { return 0; }
#else
#error unsupported platform
#endif

int checkMemory(long long gmemNeeded, long long sysmemNeeded) {
  long long gmemAvail = getDeviceMemory();
  long long sysmemAvail = getSystemMemory();
  const long long ONE_MB = 1024 * 1024;

  fprintf(stdout,
          "^^^^ gemmAvail=%lld, gmemNeeded=%lld, GMEMRESERVE=%d\n",
          gmemAvail / ONE_MB,
          gmemNeeded / ONE_MB,
          GMEM_RESERVE_MB);
  fprintf(stdout,
          "^^^^ sysmemAvail %lld, sysmemNeeded=%lld, SYSMEM_RESERVE=%d \n",
          sysmemAvail / ONE_MB,
          sysmemNeeded / ONE_MB,
          SYSMEM_RESERVE_MB);

  if ((gmemAvail - GMEM_RESERVE_MB * ONE_MB) < gmemNeeded) {
    fprintf(stdout, "^^^^ test waived: insufficient GPU memory\n");
    return 2;
  }

  if ((sysmemAvail - (long long)SYSMEM_RESERVE_MB * ONE_MB) < sysmemNeeded) {
    fprintf(stdout, "^^^^ test waived: insufficient system memory\n");
    return 2;
  }

  return 0;
}
