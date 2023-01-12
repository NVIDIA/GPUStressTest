
/**
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

/* 4/18/2020 Derived from NVIDIA internal cublasMatmulBench, http://nvbugs/200591009
** Modified to create a GPU acceptance test utility on request from Microsoft
** http://nvbugs/2772765
** Purpose: Drive all GPU present to full power, TFLOPS and memory utilization and report PASS / FAIL
** Provide a watchdog timeout to detect hung tests.
** Test time is controlled by command line option T=<loop count>
** test timeout is hardcoded to 600 seconds per test.
** exit -1 on fail.
**
*/
#include "common_header.h"
#include "command_line.h"
#include "test_args.h"
#include "test_util.h"
#include "exceptions.h"
#include "fillMatrixDevice_kernel.h"
#include "memory.h"
#include "type_convert.h"
#include "common.h"
#include <cuda_runtime.h>

/* fault injection */
#include <thread>        
#include <chrono>         

/* watchdog includes; POSIX support on Windows with:
** 
https://docs.microsoft.com/en-us/cpp/build/vcpkg?view=vs-2019
https://github.com/microsoft/vcpkg.git
**
*/
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>

/* GST specific */
#include "GST.h"

extern bool parse_in_math_scale_out_type(BlasOpts &blas_opts, const string &in_math_scale_out_type);

/* Test metadata for watchdog oversight -------------------------------------------------------------- */
struct test_state {
    const char* test_name;
    int test_state; /* 1 is running, 0 is not running */
    time_t start_time;
    time_t end_time;
    int dummy;
};

/*Globals used by test meta data: TODO needs clean up:
** wrap in meta test class
*/
bool has_error = false;
bool test_ran = true;
struct test_state tstate[NUM_TESTS];
bool tests_done = false;
bool test_hung = false;
bool watchdog_bailed = false;

/*
 * Semaphores used for watchdog implementation:
 * wd - main() to watchdog to indicate test started
 * go - watchdog to main() to indicate run next test
 * done - main() to watchdog to indicate test complete 
 */
sem_t wd, go, done;


void reset_blas_opts(CommandLine& command_line, BlasOpts& blas_opts);

void* watchdog(void* in)
{
    printf("WATCHDOG starting, TIMEOUT: %d seconds\n", TEST_WAIT_TIME);

    int i = 0, n = 0;
    struct timespec ts;
    
    sem_post(&go);
    do {
        sem_wait(&wd);

        auto now = std::chrono::system_clock::now();
        auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
        auto epoch_secs = secs.time_since_epoch();
        auto value_secs = std::chrono::duration_cast<std::chrono::seconds>(epoch_secs);
        ts.tv_sec = value_secs.count();
        ts.tv_nsec = 0L;
        ts.tv_sec += TEST_WAIT_TIME;
        n = sem_timedwait(&done, &ts);
        if ((n == -1) && (errno == ETIMEDOUT) && (tstate[i].test_state == 1)) {
            printf("TEST %s appears to be hung\n", tstate[i].test_name);
            printf("Terminating stress testing...\n");
            test_hung = true;
            sem_post(&go);
            break;
        }
        else if (n == -1) {
            perror("WATCHDOG sem_timedwait\n");
            printf("WATCHDOG thread exiting....\n");
            watchdog_bailed = true;
            pthread_exit(NULL);
        }
        sem_post(&go);
        i++;
    } while ((tests_done != true) || (test_hung != true));

    printf("WATCHDOG thread exiting....\n");
    pthread_exit(NULL);

    return(NULL);
}
/* ---------------------------------------------------------------------------------------------------------------------------*/

/*The base code for GST is cublasMatMulbench which accepts command 
**line arguments largely ignored by GST but left intact. Existing
** options include the time_loop "-T=<loop count>" which is used by GST
** and defaults to 100 requiring a runtime of around 30 min for five tests
** on a V100 for reference ad drives the GPU to full power, TFLOPS and memory
*/


using namespace std::chrono;
using cublas::CommandLine;

static float median(std::vector<float> &times) {
  const size_t size = times.size();
  if (size == 0) {
    return 0;
  }

  std::sort(times.begin(), times.end());

  const size_t mid = size / 2;
  if (size % 2 == 0) {
    return (times[mid] + times[mid - 1]) / 2;
  } else {
    return times[mid];
  }
}

template <typename T_IN_A, typename T_IN_B, typename T_IN_C, typename T_OUT,
          typename T_MATH, typename T_SCALE>
static double calc_matmul_perf_time(
    const BlasOpts &blas_opts, cublasLtHandle_t ltHandle,
    cublasLtMatmulDesc_t matmulDesc, const T_SCALE *alpha, const T_IN_A *A,
    cublasLtMatrixLayout_t Adesc, const T_IN_B *B, cublasLtMatrixLayout_t Bdesc,
    const T_SCALE *beta, T_IN_C *C, cublasLtMatrixLayout_t Cdesc, T_OUT *D,
    cublasLtMatrixLayout_t Ddesc, cublasLtMatmulAlgo_t *algo, void *workspace,
    size_t workspaceSize) {
  struct DurationEvent {
    void Start() { start = high_resolution_clock::now(); }

    void Stop() { stop = high_resolution_clock::now(); }

    double GetDuration() const {
      return duration_cast<duration<double>>(stop - start).count();
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point stop;
  };

  struct CudaDurationEvent {
    CudaDurationEvent() {
      cublas::cuda_check_error(cudaEventCreate(&start),
                               "cudaEventCreate for start failed");
      cublas::cuda_check_error(cudaEventCreate(&stop),
                               "cudaEventCreate for stop failed");
    }

    ~CudaDurationEvent() {
      // don't throw from destructor, this is often called during unwind
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    void Start(cudaStream_t stream) {
      cublas::cuda_check_error(cudaEventRecord(start, stream),
                               "cudaEventRecord for start failed");
    }

    void Stop(cudaStream_t stream) {
      cublas::cuda_check_error(cudaEventRecord(stop, stream),
                               "cudaEventRecord for stop failed");
    }

    float GetDuration() const {
      float duration;
      cublas::cuda_check_error(cudaEventElapsedTime(&duration, start, stop),
                               "cudaEventElapsedTime failed");
      return duration;
    }

   private:
    cudaEvent_t start;
    cudaEvent_t stop;
  };

  struct SemaphoreControls {
    SemaphoreControls(bool allocate = false) {
      if (allocate) {
        cublas::cuda_check_error(
            gpuAllocPinnedAndMap(sizeof(*waitCtrHostPtr),
                                 &reinterpret_cast<void *&>(waitCtrHostPtr),
                                 &reinterpret_cast<void *&>(waitCtrDevicePtr)),
            "gpuAllocPinnedAndMap failed");
      }
    }

    ~SemaphoreControls() { cudaFreeHost(waitCtrHostPtr); }

    int32_t *waitCtrHostPtr = nullptr;
    int32_t *waitCtrDevicePtr = nullptr;
  };

  SemaphoreControls semaphoreControls(true);

  cublas::cuda_check_error(cudaDeviceSynchronize(),
                           "cudaDeviceSynchronize failed");

  int warmup_loops = 1;

  wait_kernel_set_semaphore(semaphoreControls.waitCtrHostPtr, 0);

  const int timingEventsCount = blas_opts.timing_loop;
  const int apiTimingEventsCount = blas_opts.timing_loop + warmup_loops;
  std::vector<CudaDurationEvent> timingEvents(timingEventsCount);
  std::vector<DurationEvent> apiTimingEvents(apiTimingEventsCount);

  {
    for (int loop = 0; loop < apiTimingEventsCount; loop++) {
      if (loop >= warmup_loops) {
        wait_kernel(semaphoreControls.waitCtrDevicePtr, loop + 1);
        timingEvents[loop - warmup_loops].Start(0);
      }

      const size_t baseIndex = loop;

      apiTimingEvents[baseIndex].Start();
      cublas::cublas_check_error(
          cublasLtMatmul(ltHandle, matmulDesc, alpha, A, Adesc, B, Bdesc, beta,
                         C, Cdesc, D, Ddesc, algo, workspace, workspaceSize, 0),
          "cublasLtMatmul failed");
      apiTimingEvents[baseIndex].Stop();

      if (loop >= warmup_loops) {
        timingEvents[loop - warmup_loops].Stop(0);
        wait_kernel_set_semaphore(semaphoreControls.waitCtrHostPtr, loop + 1);
      }
    }

    cublas::cuda_check_error(cudaStreamSynchronize(0),
                             "cudaStreamSynchronize failed");
  }

  cublas::cuda_check_error(cudaDeviceSynchronize(),
                           "cudaDeviceSynchronize failed");

  double sum = 0;
  const size_t size = timingEvents.size();
  std::vector<double> times(size);
  for (size_t i = 0; i < size; ++i) {
    auto v = timingEvents[i].GetDuration();
    times[i] = v;
    sum += v;
  }

  std::sort(std::begin(times), std::end(times));

  auto get_percentile = [&](double p) {
    int index = std::max<int>(0, (int)std::round((double)size * p) - 1);
    return times[index];
  };

  auto min = times.front();
  auto max = times.back();
  auto percentile20 = get_percentile(0.2);
  auto percentile50 = get_percentile(0.5);
  // average throws away the two extreme values
  auto calc_mean = [&] {
    if (size > 2) {
      sum -= min + max;
      return sum / (static_cast<double>(size) - 2);
    } else {
      return sum / static_cast<double>(size);
    }
  };

  auto mean = calc_mean();

  double kernel_time = mean;

  fprintf(stdout,
          "^^^^ gpu time statistics: runs %d, mean %f ms, min %f ms, 20 "
          "percent %f ms, "
          "50 percent %f ms, max "
          "%f ms\n",
          blas_opts.timing_loop, (double)mean, (double)min,
          (double)percentile20, (double)percentile50, (double)max);

  double cudaTime = kernel_time * size / 1000.;
  return cudaTime;
}

template <typename T_IN_A, typename T_IN_B, typename T_IN_C, typename T_OUT,
          typename T_MATH, typename T_SCALE>
static void auto_tuning(cublasLtHandle_t ltHandle,
                        cublasLtMatmulDesc_t computeDesc, const T_SCALE *alpha,
                        const T_IN_A *A, cublasLtMatrixLayout_t Adesc,
                        const T_IN_B *B, cublasLtMatrixLayout_t Bdesc,
                        const T_SCALE *beta, T_IN_C *C,
                        cublasLtMatrixLayout_t Cdesc, T_OUT *D,
                        cublasLtMatrixLayout_t Ddesc, void *workspace,
                        size_t workspaceSize, cublasLtMatmulAlgo_t &algo) {
  cublasLtMatmulPreference_t pref = NULL;
  const int requested_algo_count = 8;
  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result[requested_algo_count];
  int best_algo_idx = 0;
  float time = 0;
  float best_algo_time = 0;
  cudaStream_t stream;
  cudaEvent_t start_event, stop_event;

  cublas::cublas_check_error(cublasLtMatmulPreferenceCreate(&pref),
                             "create preference failed");
  cublas::cublas_check_error(cublasLtMatmulPreferenceSetAttribute(
                                 pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                 &workspaceSize, sizeof(workspaceSize)),
                             "set preference workspace failed");

  cublas::cublas_check_error(
      cublasLtMatmulAlgoGetHeuristic(ltHandle, computeDesc, Adesc, Bdesc, Cdesc,
                                     Ddesc, pref, requested_algo_count,
                                     heuristic_result, &returned_results),
      "cublasLtMatmulAlgoGetHeuristic failed");

  if (returned_results == 0) {
    cublas::cublas_check_error(CUBLAS_STATUS_NOT_SUPPORTED, "no algo found");
  }
  cublas::cuda_check_error(cudaStreamCreate(&stream), "create stream failed");
  cublas::cuda_check_error(cudaEventCreate(&start_event),
                           "create start event failed");
  cublas::cuda_check_error(cudaEventCreate(&stop_event),
                           "create stop event failed");

  constexpr int repeat_algo_check = 5;
  std::vector<float> algo_times(repeat_algo_check);

  for (int algo_idx = 0; algo_idx < returned_results; ++algo_idx) {
    for (int check_idx = 0; check_idx < repeat_algo_check; ++check_idx) {
      cublas::cuda_check_error(cudaEventRecord(start_event, stream),
                               "cudaEventRecord for start_event failed");
      cublas::cublas_check_error(
          cublasLtMatmul(ltHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta,
                         C, Cdesc, D, Ddesc, &heuristic_result[algo_idx].algo,
                         workspace, workspaceSize, stream),
          "cublasLtMatmul failed in auto_tuning");
      cublas::cuda_check_error(cudaEventRecord(stop_event, stream),
                               "cudaEventRecord for stop_event failed");
      cublas::cuda_check_error(cudaEventSynchronize(stop_event),
                               "cudaEventSynchronize for stop_event failed");
      cublas::cuda_check_error(
          cudaEventElapsedTime(&time, start_event, stop_event),
          "cudaEventElapsedTime failed");
      algo_times[check_idx] = time;
    }

    time = median(algo_times);
    if ((algo_idx == 0) || (time < best_algo_time)) {
      best_algo_time = time;
      best_algo_idx = algo_idx;
    }
  }

  memcpy(&algo, &heuristic_result[best_algo_idx].algo, sizeof(algo));

  if (pref) {
    cublas::cublas_check_error(cublasLtMatmulPreferenceDestroy(pref),
                               "cublasLtMatmulPreferenceDestroy failed");
  }

  if (stream) {
    cublas::cuda_check_error(cudaStreamDestroy(stream),
                             "cudaStreamDestroy for stream failed");
  }

  if (start_event) {
    cublas::cuda_check_error(cudaEventDestroy(start_event),
                             "cudaEventDestroy for start_event failed");
  }

  if (stop_event) {
    cublas::cuda_check_error(cudaEventDestroy(stop_event),
                             "cudaEventDestroy for stop_event failed");
  }
}

template <typename T_IN_A, typename T_IN_B, typename T_IN_C, typename T_OUT,
          typename T_MATH, typename T_SCALE>
static int lt_gemm(cublasLtHandle_t ltHandle, const BlasOpts &blas_opts,
                   T_IN_A *A, T_IN_B *B, T_IN_C *C, T_OUT *D, T_SCALE alpha,
                   T_SCALE beta, int lda, int ldb, int ldc) {
  try {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    void *workspace = nullptr;
    using BiasType =
        typename biasTypeExtended<T_IN_A, T_IN_B, T_IN_C, T_OUT, T_SCALE>::type;
    BiasType *Bias = nullptr;
    int ldatransform =
        blas_opts.m_orderingA == CUBLASLT_ORDER_COL ? lda : 32 * lda;
    int ldbtransform = 0;
    int ldctransform =
        blas_opts.m_orderingC == CUBLASLT_ORDER_COL ? ldc : 32 * ldc;

    int device_version = 0;
    cublas::cuda_check_error(get_device_version(device_version),
                             "get device version failed");
    // 4MB on prior architectures and 32MB on Hopper
    size_t workspaceSize =
        (device_version < 900) ? 1024 * 1024 * 4 : 1024 * 1024 * 32;

    switch (blas_opts.m_orderingB) {
      case CUBLASLT_ORDER_COL32_2R_4R4:  // for ampere
        ldbtransform = 32 * roundoff(ldb, 32);
        break;
      case CUBLASLT_ORDER_COL:
        ldbtransform = ldb;
        break;
      default:
        ldbtransform = 32 * roundoff(ldb, 8);
        break;
    }

    cublas::cuda_check_error(cudaMalloc(&workspace, workspaceSize),
                             "cudaMalloc for workspace failed");

    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL,
                           CtransformDesc = NULL, DtransformDesc = NULL;

    cublas::cublas_check_error(
        cublasLtMatmulDescCreate(&matmulDesc, blas_opts.compute_type,
                                 blas_opts.scale_type),
        "create MatmulDesc failed");
    cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(
                                   matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &blas_opts.transb, sizeof(blas_opts.transb)),
                               "set DESC_TRANSB failed");
    cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(
                                   matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &blas_opts.transa, sizeof(blas_opts.transa)),
                               "set DESC_TRANSA failed");
    cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(
                                   matmulDesc, CUBLASLT_MATMUL_DESC_TRANSC,
                                   &blas_opts.transc, sizeof(blas_opts.transc)),
                               "set DESC_TRANSC failed");
    cublas::cublas_check_error(
        cublasLtMatmulDescSetAttribute(
            matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &blas_opts.m_epilogue,
            sizeof(blas_opts.m_epilogue)),
        "set DESC_EPILOGUE failed");
    if (blas_opts.m_epilogue & CUBLASLT_EPILOGUE_BIAS) {
      cublas::cuda_check_error(
          cudaMalloc(&Bias, blas_opts.m * sizeof(BiasType)),
          "cudaMalloc for Bias failed");
      if (!blas_opts.filling_zero) {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(Bias, blas_opts.m),
                             blas_opts.m, blas_opts.m, blas_opts.m, 1,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             0, 0, 0, true, 1),
            "fillMatrixDevice Bias failed");
      } else {
        cublas::cuda_check_error(
            cudaMemset(Bias, 0, blas_opts.m * sizeof(BiasType)),
            "cudaMemset for Bias failed");
      }
      cublas::cublas_check_error(
          cublasLtMatmulDescSetAttribute(matmulDesc,
                                         CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                         &Bias, sizeof(Bias)),
          "set DESC_BIAS_POINTER failed");
    }
    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    cublas::cublas_check_error(
        cublasLtMatrixLayoutCreate(
            &AtransformDesc, blas_opts.input_type_a,
            blas_opts.transa == CUBLAS_OP_N ? blas_opts.m : blas_opts.k,
            blas_opts.transa == CUBLAS_OP_N ? blas_opts.k : blas_opts.m,
            ldatransform),
        "create MatrixLayout for AtransformDesc failed");
    cublas::cublas_check_error(
        cublasLtMatrixLayoutSetAttribute(
            AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &blas_opts.m_orderingA, sizeof(blas_opts.m_orderingA)),
        "set LAYOUT_ORDER for AtransformDesc failed");
    cublas::cublas_check_error(
        cublasLtMatrixLayoutCreate(
            &BtransformDesc, blas_opts.input_type_b,
            blas_opts.transb == CUBLAS_OP_N ? blas_opts.k : blas_opts.n,
            blas_opts.transb == CUBLAS_OP_N ? blas_opts.n : blas_opts.k,
            ldbtransform),
        "create MatrixLayout for BtransformDesc failed");
    cublas::cublas_check_error(
        cublasLtMatrixLayoutSetAttribute(
            BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &blas_opts.m_orderingB, sizeof(blas_opts.m_orderingB)),
        "set LAYOUT_ORDER for BtransformDesc failed");

    cublas::cublas_check_error(
        cublasLtMatrixLayoutCreate(&CtransformDesc, blas_opts.input_type_c,
                                   blas_opts.m, blas_opts.n, ldctransform),
        "create MatrixLayout for CtransformDesc failed");
    cublas::cublas_check_error(
        cublasLtMatrixLayoutSetAttribute(
            CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
            &blas_opts.m_orderingC, sizeof(blas_opts.m_orderingC)),
        "set LAYOUT_ORDER for CtransformDesc failed");
    if (blas_opts.m_outOfPlace) {
      cublas::cublas_check_error(
          cublasLtMatrixLayoutCreate(&DtransformDesc, blas_opts.output_type,
                                     blas_opts.m, blas_opts.n, ldctransform),
          "create MatrixLayout for DtransformDesc failed");
      cublas::cublas_check_error(
          cublasLtMatrixLayoutSetAttribute(
              DtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
              &blas_opts.m_orderingC, sizeof(blas_opts.m_orderingC)),
          "set LAYOUT_ORDER for DtransformDesc failed");
    } else {
      DtransformDesc = CtransformDesc;
    }

    cublasLtMatmulAlgo_t algo;
    if (blas_opts.quick_autotuning) {
      auto_tuning<T_IN_A, T_IN_B, T_IN_C, T_OUT, T_MATH, T_SCALE>(
          ltHandle, matmulDesc, static_cast<const T_SCALE *>(&alpha),
          static_cast<const T_IN_A *>(A), AtransformDesc,
          static_cast<const T_IN_B *>(B), BtransformDesc,
          static_cast<const T_SCALE *>(&beta), C, CtransformDesc, D,
          DtransformDesc, workspace, workspaceSize, algo);
    }
    // ---------------------------------------------------------------------------------------------
    // computation
    char ta = operation_to_char(blas_opts.transa);
    char tb = operation_to_char(blas_opts.transb);
    printf("#### args: ta=%c tb=%c m=%d n=%d k=%d ", ta, tb, blas_opts.m,
           blas_opts.n, blas_opts.k);
    printCuType(" alpha =", alpha);
    printCuType(" beta=", beta);
    printf("\n");
    printf("#### args: lda=%d ldb=%d ldc=%d ldd=%d loop=%d\n", ldatransform,
           ldbtransform, ldctransform, ldctransform, blas_opts.timing_loop);

    double cudaTime =
        calc_matmul_perf_time<T_IN_A, T_IN_B, T_IN_C, T_OUT, T_MATH, T_SCALE>(
            blas_opts, ltHandle, matmulDesc,
            static_cast<const T_SCALE *>(&alpha),
            static_cast<const T_IN_A *>(A), AtransformDesc,
            static_cast<const T_IN_B *>(B), BtransformDesc,
            static_cast<const T_SCALE *>(&beta), C, CtransformDesc, D,
            DtransformDesc, blas_opts.quick_autotuning ? &algo : NULL,
            workspace, workspaceSize);

    double flopsCoef = 2.0;

    if ((blas_opts.math_type == CUDA_C_32F) ||
        (blas_opts.math_type == CUDA_C_64F)) {
      flopsCoef = 8.0;
    }

    double TheoreticalFlops = flopsCoef * (double)blas_opts.m *
                              (double)blas_opts.n * (double)blas_opts.k;
    double cudaGflops =
        blas_opts.timing_loop * (1e-9 * TheoreticalFlops) / cudaTime;
    cublasPrintPerf(false, cudaTime, cudaGflops);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (blas_opts.m_outOfPlace) {
      if (DtransformDesc) {
        cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(DtransformDesc),
                                   "destory DtransformDesc failed");
      }
    }
    if (CtransformDesc)
      cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(CtransformDesc),
                                 "destory CtransformDesc failed");
    if (BtransformDesc)
      cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(BtransformDesc),
                                 "destory BtransformDesc failed");
    if (AtransformDesc)
      cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(AtransformDesc),
                                 "destory AtransformDesc failed");
    if (matmulDesc)
      cublas::cublas_check_error(cublasLtMatmulDescDestroy(matmulDesc),
                                 "destroy matmulDesc failed");
    if (workspace)
      cublas::cuda_check_error(cudaFree(workspace), "free workspace failed");
    if (Bias) cublas::cuda_check_error(cudaFree(Bias), "free Bias failed");
  } catch (cublas::cuda_exception &e) {
    cout << e << endl;
    return 1;
  } catch (cublas::cublas_exception &e) {
    cout << e << endl;
    return 1;
  } catch (const std::exception &e) {
    cout << e.what() << endl;
    return 1;
  }

  return 0;
}

template <typename T_IN_A, typename T_IN_B, typename T_IN_C, typename T_OUT,
          typename T_MATH, typename T_SCALE>
static void test_engine(BlasOpts &blas_opts) {
  printf("testing cublasLt\n");
  try {
    T_IN_A *d_A = nullptr;
    T_IN_B *d_B = nullptr;
    T_IN_C *d_C = nullptr;
    T_OUT *d_D = nullptr;
    T_SCALE alpha = cuGet<T_SCALE>(blas_opts.alpha);
    T_SCALE beta = cuGet<T_SCALE>(blas_opts.beta);
    int matrixM = 0, matrixN = 0, matrixK = 0;
    int rowsA = 0, rowsB = 0, rowsC = 0;
    int colsA = 0, colsB = 0, colsC = 0;
    size_t matrixSizeA = 0, matrixSizeB = 0, matrixSizeC = 0;

    // make sure no error
    if (!std::is_same<T_IN_C, T_OUT>::value) {
      blas_opts.m_outOfPlace = true;
    }

    matrixM = blas_opts.m;
    matrixN = blas_opts.n;
    matrixK = blas_opts.k;

    if (blas_opts.lda) {
      if ((blas_opts.transa == CUBLAS_OP_N) && (blas_opts.lda < matrixM)) {
        fprintf(stdout, "lda(=%d) must be bigger than m(=%d)\n", blas_opts.lda,
                matrixM);
        return;
      }
      if ((blas_opts.transa != CUBLAS_OP_N) && (blas_opts.lda < matrixK)) {
        fprintf(stdout, "lda(=%d) must be bigger than k(=%d) for ta\n",
                blas_opts.lda, matrixK);
        return;
      }
    }
    if (blas_opts.ldb) {
      if ((blas_opts.transb == CUBLAS_OP_N) && (blas_opts.ldb < matrixK)) {
        fprintf(stdout, "ldb(=%d) must be bigger than k(=%d)\n", blas_opts.ldb,
                matrixK);
        return;
      }
      if ((blas_opts.transb != CUBLAS_OP_N) && (blas_opts.ldb < matrixN)) {
        fprintf(stdout, "ldb(=%d) must be bigger than n(=%d) for tb\n",
                blas_opts.ldb, matrixN);
        return;
      }
    }
    if ((blas_opts.ldc) && (blas_opts.ldc < matrixM)) {
      fprintf(stdout, "ldc(=%d) must be bigger than m(=%d)\n", blas_opts.ldc,
              matrixM);
      return;
    }

    if (blas_opts.transa != CUBLAS_OP_N) {
      rowsA = imax(blas_opts.lda, matrixK);
      colsA = matrixM;
    } else {
      rowsA = imax(blas_opts.lda, matrixM);
      colsA = matrixK;
    }
    if (blas_opts.transb != CUBLAS_OP_N) {
      rowsB = imax(blas_opts.ldb, matrixN);
      colsB = matrixK;
    } else {
      rowsB = imax(blas_opts.ldb, matrixK);
      colsB = matrixN;
    }
    rowsC = imax(blas_opts.ldc, matrixM);
    colsC = matrixN;

    matrixSizeA = (size_t)rowsA * colsA;
    matrixSizeB = (size_t)rowsB * colsB;
    matrixSizeC = (size_t)rowsC * colsC;

    printf("matrixSize Total: %ld \n", matrixSizeA +  matrixSizeB + matrixSizeC);

    d_A = cublas::device_memory::allocate<T_IN_A>(matrixSizeA);
    d_B = cublas::device_memory::allocate<T_IN_B>(matrixSizeB);
    d_C = cublas::device_memory::allocate<T_IN_C>(matrixSizeC);
    d_D = blas_opts.m_outOfPlace
              ? cublas::device_memory::allocate<T_OUT>(matrixSizeC)
              : (T_OUT *)d_C;

    if (!blas_opts.filling_zero) {
      if (blas_opts.transa != CUBLAS_OP_N) {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(d_A, matrixSizeA),
                             matrixSizeA, rowsA, blas_opts.k, blas_opts.m,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             0, 0, 0, true,
                             1  // only use first buffer, setMatricesForGEMM
                                // will copy to the others
                             ),
            "fillMatrixDevice for matrix A failed");
      } else {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(d_A, matrixSizeA),
                             matrixSizeA, rowsA, blas_opts.m, blas_opts.k,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             0, 0, 0, true,
                             1  // only use first buffer, setMatricesForGEMM
                                // will copy to the others
                             ),
            "fillMatrixDevice for matrix A failed");
      }

      if (blas_opts.transb != CUBLAS_OP_N) {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(d_B, matrixSizeB),
                             matrixSizeB, rowsB, blas_opts.n, blas_opts.k,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             121, 0, 0, true,
                             1  // only use first buffer, setMatricesForGEMM
                                // will copy to the others
                             ),
            "fillMatrixDevice for matrix B failed");
      } else {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(d_B, matrixSizeB),
                             matrixSizeB, rowsB, blas_opts.k, blas_opts.n,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             121, 0, 0, true,
                             1  // only use first buffer, setMatricesForGEMM
                                // will copy to the others
                             ),
            "fillMatrixDevice for matrix B failed");
      }

      cublas::cuda_check_error(
          fillMatrixDevice(make_bufferBatchVariant(d_C, matrixSizeC),
                           matrixSizeC, rowsC, blas_opts.m, blas_opts.n,
                           CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P', 0,
                           0, 0, true, 1),
          "fillMatrixDevice for matrix C failed");
      if (blas_opts.m_outOfPlace) {
        cublas::cuda_check_error(
            fillMatrixDevice(make_bufferBatchVariant(d_D, matrixSizeC),
                             matrixSizeC, rowsC, blas_opts.m, blas_opts.n,
                             CUBLAS_FILL_MODE_FULL, CUBLAS_DIAG_NON_UNIT, 'P',
                             0, 0, 0, true, 1),
            "fillMatrixDevice for matrix D failed");
      }
    } else {
      cublas::cuda_check_error(cudaMemset(d_A, 0, sizeof(T_IN_A) * matrixSizeA),
                               "cudaMemset for matrix A failed");
      cublas::cuda_check_error(cudaMemset(d_B, 0, sizeof(T_IN_B) * matrixSizeB),
                               "cudaMemset for matrix B failed");
      cublas::cuda_check_error(cudaMemset(d_C, 0, sizeof(T_IN_C) * matrixSizeC),
                               "cudaMemset for matrix C failed");
      if (blas_opts.m_outOfPlace) {
        cublas::cuda_check_error(
            cudaMemset(d_D, 0, sizeof(T_OUT) * matrixSizeC),
            "cudaMemset for matrix D failed");
      }
    }

    cublasLtHandle_t ltHandle;
    cublas::cublas_check_error(cublasLtCreate(&ltHandle),
                               "create cublasLt handle failed");

    bool has_error = false;
    if (lt_gemm<T_IN_A, T_IN_B, T_IN_C, T_OUT, T_MATH, T_SCALE>(
            ltHandle, blas_opts, d_A, d_B, d_C, d_D, alpha, beta, rowsA, rowsB,
            rowsC)) {
      has_error = true;
    }

    cublas::device_memory::free(d_A);
    cublas::device_memory::free(d_B);
    cublas::device_memory::free(d_C);
    if (blas_opts.m_outOfPlace) {
      cublas::device_memory::free(d_D);
    }
    cublas::cublas_check_error(cublasLtDestroy(ltHandle),
                               "destroy ltHandle failed");

    if (has_error) {
      printf("testing cublasLt fail\n");
      exit(-1);
    } else {
      printf("testing cublasLt pass\n");
    }

  } catch (cublas::cuda_exception &e) {
    cout << e << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (cublas::cublas_exception &e) {
    cout << e << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (const std::exception &e) {
    cout << e.what() << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  }
}

#define TEST_ENGINE_MAPPING(T_IN_A, T_IN_B, T_IN_C, T_OUT, T_SCALE, T_MATH)    \
  if ((blas_opts.input_type_a == T_IN_A) &&                                    \
      (blas_opts.input_type_b == T_IN_B) &&                                    \
      (blas_opts.input_type_c == T_IN_C) &&                                    \
      (blas_opts.output_type == T_OUT) && (blas_opts.scale_type == T_SCALE)) { \
    test_engine<typename CudaTypeEnumTraits<T_IN_A>::type,                     \
                typename CudaTypeEnumTraits<T_IN_B>::type,                     \
                typename CudaTypeEnumTraits<T_IN_C>::type,                     \
                typename CudaTypeEnumTraits<T_OUT>::type,                      \
                typename CudaTypeEnumTraits<T_MATH>::type,                     \
                typename CudaTypeEnumTraits<T_SCALE>::type>(blas_opts);        \
  }


static void test_cublasLt(BlasOpts &blas_opts) {
  int device_version = 0;
  cublas::cuda_check_error(get_device_version(device_version),
                           "get device version failed");
  try {
    switch (blas_opts.math_type) {
      case CUDA_R_32F: {
        if ((blas_opts.input_type_a == CUDA_R_8F_E4M3) &&
            (device_version < 900)) {
          printf("not supported for the FP8 options\n");
          return;
        }
        // sss A,B : FP32 ->  C FP32
        TEST_ENGINE_MAPPING(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                            CUDA_R_32F, CUDA_R_32F)
        // hss A,B FP16 ->  C FP32
        TEST_ENGINE_MAPPING(CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F,
                            CUDA_R_32F, CUDA_R_32F)
        // hsh A,B FP16 ->  C FP16
        TEST_ENGINE_MAPPING(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                            CUDA_R_32F, CUDA_R_32F)
        // qqssq A,B:fp8_e4m3, C:bfloat16, scale type: float, output type:
        // fp8_e4m3
        TEST_ENGINE_MAPPING(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF,
                            CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F)
        // tss A,B : BF16 ->  C FP32
        TEST_ENGINE_MAPPING(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F,
                            CUDA_R_32F, CUDA_R_32F)
        // tst A,B : BF16 ->  C BF16
        TEST_ENGINE_MAPPING(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                            CUDA_R_32F, CUDA_R_32F)
      } break;
      case CUDA_C_32F:  // ccc
        TEST_ENGINE_MAPPING(CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F,
                            CUDA_C_32F, CUDA_C_32F)
        break;
      case CUDA_R_64F:  // ddd A,B : FP64 ->  C FP64
        TEST_ENGINE_MAPPING(CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
                            CUDA_R_64F, CUDA_R_64F)
        break;
      case CUDA_C_64F:  // zzz
        TEST_ENGINE_MAPPING(CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F,
                            CUDA_C_64F, CUDA_C_64F)
        break;
      case CUDA_R_16F:  // hhh
        TEST_ENGINE_MAPPING(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                            CUDA_R_16F, CUDA_R_16F)
        break;
      case CUDA_R_32I: {
        if (device_version < 750) {
          printf("not supported for the imma options\n");
          return;
        }
        if (blas_opts.transa == CUBLAS_OP_N) {  // make sure NT
          blas_opts.m_orderingA = CUBLASLT_ORDER_COL32;
          blas_opts.m_orderingB = device_version >= 800
                                      ? CUBLASLT_ORDER_COL32_2R_4R4
                                      : CUBLASLT_ORDER_COL4_4R2_8C;
          blas_opts.m_orderingC = CUBLASLT_ORDER_COL32;
          blas_opts.transa = CUBLAS_OP_N;
          blas_opts.transb = CUBLAS_OP_T;
        } else {  // make sure TN
          blas_opts.transa = CUBLAS_OP_T;
          blas_opts.transb = CUBLAS_OP_N;
        }
        // bisb_imma
        TEST_ENGINE_MAPPING(CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I,
                            CUDA_R_32F, CUDA_R_32I)
        // bii_imma
        TEST_ENGINE_MAPPING(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
                            CUDA_R_32I, CUDA_R_32I)
      } break;
      default:
        printf("mode not supported\n");
        break;
    }
  } catch (cublas::cuda_exception &e) {
    cout << e << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (const std::exception &e) {
    cout << e.what() << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  }
}

/* ------------------------------------------------------------------------------------------------------------------------------- */

int main(int argc, char *argv[]) {
  int ret = 0;
  pthread_t wd_thread;
  pthread_attr_t attr;
  GST gst;

  sem_init(&wd, 0, 0);
  sem_init(&go, 0, 0);
  sem_init(&done, 0, 0);
  void(*watchdog(void*));

  if (pthread_attr_init(&attr)) {
    perror("pthread_attr_init - watchdog");
#define TEST_ENGINE_MAPPING(T_IN_A, T_IN_B, T_IN_C, T_OUT, T_SCALE, T_MATH)    \
  if ((blas_opts.input_type_a == T_IN_A) &&                                    \
      (blas_opts.input_type_b == T_IN_B) &&                                    \
      (blas_opts.input_type_c == T_IN_C) &&                                    \
      (blas_opts.output_type == T_OUT) && (blas_opts.scale_type == T_SCALE)) { \
    test_engine<typename CudaTypeEnumTraits<T_IN_A>::type,                     \
                typename CudaTypeEnumTraits<T_IN_B>::type,                     \
                typename CudaTypeEnumTraits<T_IN_C>::type,                     \
                typename CudaTypeEnumTraits<T_OUT>::type,                      \
                typename CudaTypeEnumTraits<T_MATH>::type,                     \
                typename CudaTypeEnumTraits<T_SCALE>::type>(blas_opts);        \
  }
    exit(-1);
  }

  if (pthread_create(&wd_thread, &attr, watchdog, (void*)NULL) != 0) {
      perror("pthread create - watchdog");
      exit(-1);
  }

  printf("%s capturing GPU information...\n", argv[0]);

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
	printf("Exiting...\n");
	exit(-1);
  } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  CommandLine command_line(argc, argv);

  int device_arg = -1;
  if (command_line.check_cmd_line_flag("dv")) {
    command_line.get_cmd_line_argument("dv", device_arg);
    // arg check 1
    if (device_arg > deviceCount) {
      printf("Device (dv) #%d parameter is too big\n", device_arg);
      exit(1);
    }
    // arg check 2
    if (device_arg < 0) {
      printf("Device (dv) #%d parameter is too small\n", device_arg);
      exit(1);
    }
    printf("Device #%d is selected\n", device_arg);
  }

  BlasOpts blas_opts;
  parse_args(command_line, blas_opts);
  reset_blas_opts(command_line, blas_opts);

  /* GPU detection and test initilization */
  int dev;
  size_t gpumem = 0LL;
  cudaDeviceProp devprops[MAX_NUM_GPUS] {};
  for (dev = 0; dev < deviceCount; dev++) {
      if ((device_arg >= 0)  && (device_arg != dev)) {
        printf("Device %d: skiped\n", dev);
        continue;
      }
      CHECK(cudaSetDevice(dev));
      CHECK(cudaGetDeviceProperties(&devprops[dev], dev));
      printf("Device %d: \"%s\"\n", dev, devprops[dev].name);
      if (dev == 0)
          gpumem = devprops[dev].totalGlobalMem;
      else {
          if (gpumem != devprops[dev - 1].totalGlobalMem) {
              printf("Detected different GPU memory sizes\n");
              printf("gpumem: %lld, GPU %d %lld\n", (long long) gpumem, (dev - 1), (long long) devprops[dev - 1].totalGlobalMem);
              printf("EXITING...\n");
              exit(0);
          }
      }
  }
  gpumem /= (1024 * 1024 * 1024);

  /* Initilize tests based on type of GPU
  */
  int memgb = 0;

#ifndef DEBUG_MATRIX_SIZES
  string gpu_name(devprops[0].name);
#else
// These entries should match GST::test_suite; clever C++ way to range over the enum and cast to string not obvious...
for (string gpu_name :  {"T4", "A100_40", "A100_80", "K80", "M60", "P40", "P100", "H100", "V100_16", "V100_32", "Generic", "NVIDIA Graphics Device"}) {

if (!gpu_name.compare(string("A100_80"))) { 
    printf("set A100_80\n");
    gpumem = 80;
}
else if (!gpu_name.compare(string("V100_32"))) {
    printf("set V100_32\n");
    gpumem = 32;
}

printf("DEBUG_MATRIX_SIZES: Checking matrix size only (no CUDA execution) for: %s\n", gpu_name.c_str());
#endif

//H100 temporay fix
if (!gpu_name.compare(string("NVIDIA Graphics Device"))) {
    gpu_name = string("H100");    
}
  
  while (true) {
    if (gpu_name.find("A100", 0) != string::npos) {

        if (gpumem > 40) {
          cout << "Initilizing A100 80 GB based test suite" << endl;
          gst = GST(GST::A100_80);
          memgb = 80;
        }  else {
          cout << "Initilizing A100 40 GB based test suite" << endl;
          gst = GST(GST::A100_40);
          memgb = 40;
        }
        break;
    }
    if (gpu_name.find("T4", 0) != string::npos) {
        cout << "Initilizing T4 based test suite" << endl;
        gst = GST(GST::T4);
        memgb = 16;
        break;
    }
    if (gpu_name.find("K80", 0) != string::npos) {
        cout << "Initilizing K80 based test suite" << endl;
        gst = GST(GST::K80);
        memgb = 11;
        break;
    }
    if (gpu_name.find("M60", 0) != string::npos) {
        cout << "Initilizing M60 based test suite" << endl;
        gst = GST(GST::M60);
        memgb = 8;
        break;
    }
    if (gpu_name.find("P40", 0) != string::npos) {
        cout << "Initilizing P40 based test suite" << endl;
        gst = GST(GST::P40);
        memgb = 22;
        break;
    }
    if (gpu_name.find("P100", 0) != string::npos) {
        cout << "Initilizing P100 based test suite" << endl;
        gst = GST(GST::P100);
        memgb = 16;
        break;
    }
    if (gpu_name.find("V100", 0) != string::npos) {

        if (gpumem > 30) {
          cout << "Initilizing V100 32 GB based test suite" << endl;
          gst = GST(GST::V100_32);
          memgb = 32;
        }  else {
          cout << "Initilizing V100 16 GB based test suite" << endl;
          gst = GST(GST::V100_16);
          memgb = 16;
        }
        break;
    }
    if (gpu_name.find("H100", 0) != string::npos) {
        cout << "Initilizing H100 based test suite" << endl;
        gst = GST(GST::H100);
        memgb = 95;
        break;
    }
    cout << "Initilizing Generic test suite" << endl;
    gst = GST(GST::Generic);
    memgb = 8;
    break;
  }


  printf("GPU Memory: %lld, memgb: %d\n", (long long) gpumem, memgb);
  printf("\n\n");


  for (dev = 0; dev < deviceCount; dev++) {
	CHECK(cudaSetDevice(dev));
	printf("Device %d: \"%s\", PCIe: %x\n", dev, devprops[dev].name,devprops[dev].pciBusID);

       // gst.dump_test_args(0);

	for (int t_num = 0; t_num  < NUM_TESTS; t_num++) {

            /* Abort if watchdog has died */
            if (watchdog_bailed) {
                printf("WATCHDOG Thread exited...\n");
                printf("GPUstress terminating\n");
                exit(-1);
            }
            reset_blas_opts(command_line, blas_opts);
            /* Debug
            gst.dump_test_args(tix);
            hello_world(blas_opts, gst.stress_tests[0].P_arg);
            */

            /* Parse command line optioms */
            bool p_parse = parse_in_math_scale_out_type(blas_opts, gst.stress_tests[t_num].P_arg);
            // cout << "DEBUG:" << "after parse" << endl;
            if (!p_parse) {
                printf("p_parse failed\n");
                exit(-1);
            }
            // cout << "DEBUG:" << "set opts" << endl;

            blas_opts.m = gst.stress_tests[t_num].m_arg;
            blas_opts.n = gst.stress_tests[t_num].n_arg;
            blas_opts.k = gst.stress_tests[t_num].k_arg;
            blas_opts.m_opt = true;
            blas_opts.n_opt = true;
            blas_opts.k_opt = true;

            if (gst.stress_tests[t_num].ta_arg == 1)
                blas_opts.transa_opt = true;
            else {
                blas_opts.transa_opt = false;
                blas_opts.transa = (cublasOperation_t)0;
            }
            if (gst.stress_tests[t_num].tb_arg == 1)
                blas_opts.transb_opt = true;
            else {
                blas_opts.transb_opt = false;
                blas_opts.transb = (cublasOperation_t)0;
            }
            blas_opts.beta = 0.0f;
            blas_opts.beta_opt = true;
        
            printf("\n***** STARTING TEST %d: %s On Device %d %s\n", t_num, gst.stress_tests[t_num].test_name, dev, devprops[dev].name);
            fflush(stdout);
            tstate[t_num].test_name = gst.stress_tests[t_num].test_name;
            tstate[t_num].test_state = 1;
            tstate[t_num].start_time = time(NULL);
            // cout << "DEBUG:" << "signal wd" << endl;
        
            /* Signal watchdog test started */
            sem_post(&wd);
            // cout << "DEBUG:" << "start test" << endl;

printf("DEBUG: transa %d transb %d \n", blas_opts.transa, blas_opts.transb);

            /* Run the test */
            test_cublasLt(blas_opts);
            printf("***** TEST %s On Device %d %s\n", gst.stress_tests[t_num].test_name, dev, devprops[dev].name);

            if (!test_ran) 
                printf("***** TEST DID NOT EXECUTE *****\n\n");
            else {
                if (has_error == true || test_hung == true) {
                    printf("***** TEST FAILED ****\n\n");
                    ret = -1;
                    break;
                }
            else
#ifndef DEBUG_MATRIX_SIZES
                printf("***** TEST PASSED ****\n");
#endif
              continue;
            }
            tstate[t_num].end_time = time(NULL);
            tstate[t_num].test_state = 0;
            printf("TEST TIME: %d seconds\n",(int)(tstate[t_num].end_time - tstate[t_num].start_time));
            cudaDeviceSynchronize();
            cudaDeviceReset();

            if (t_num == NUM_TESTS)
                tests_done = true;

            /* Signal watchdog test finished*/
            sem_post(&done);

            /* wait for watchdog to signal next next test */
            sem_wait(&go);
      }
  }
#ifdef DEBUG_MATRIX_SIZES
    gpumem=0;
}
#endif
  
  exit(ret);
}






