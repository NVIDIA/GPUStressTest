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
#include "exceptions.h"
#include "memory.h"
#include "type_convert.h"
#include "common.h"
#include <cuda_runtime.h>

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

/* WGST specific */
#include "WGST.h"

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
int tests_done = 0;
int test_hung = 0;
int watchdog_bailed = 0;

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

#ifdef __linux__
        clock_gettime(CLOCK_REALTIME, &ts)
#else
        auto now = std::chrono::system_clock::now();
        auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
        auto epoch_secs = secs.time_since_epoch();
        auto value_secs = std::chrono::duration_cast<std::chrono::seconds>(epoch_secs);
        ts.tv_sec = value_secs.count();
        ts.tv_nsec = 0L;
#endif
        ts.tv_sec += TEST_WAIT_TIME;
        n = sem_timedwait(&done, &ts);
        if ((n == -1) && (errno == ETIMEDOUT) && (tstate[i].test_state == 1)) {
            printf("TEST %s appears to be hung\n", tstate[i].test_name);
            printf("Terminating stress testing...\n");
            exit(-1);
        }
        else if ((n == 0) && (tstate[i].test_state == 0)) {
            printf("TEST %d, %s DONE\n", i, tstate[i].test_name);
        }
        else if (n == -1) {
            perror("WATCHDOG sem_timedwait\n");
            printf("WATCHDOG thread exiting....\n");
            watchdog_bailed = 1;
            pthread_exit(NULL);
        }
        sem_post(&go);
        i++;
    } while ((tests_done != 1) || (test_hung != 1));

    printf("WATCHDOG thread exiting....\n");
    pthread_exit(NULL);

    return(NULL);
}
/* ---------------------------------------------------------------------------------------------------------------------------*/

/*The base code for WGST is cublasMatMulbench which accepts command 
**line arguments largely ignored by WGST but left intact. Existing
** options include the time_loop "-T=<loop count>" which is used by WGST
** and defaults to 100 requiring a runtime of around 30 min for five tests
** on a V100 for reference ad drives the GPU to full power, TFLOPS and memory
*/
using cublas::CommandLine;

/* From base code: cublasMatmulBench */
template <typename T_IN, typename T_OUT, typename T_MATH, typename T_SCALE>
static int 
lt_gemm(cublasLtHandle_t ltHandle,
        const BlasOpts& blas_opts,
        T_IN *A,
        T_IN *B,
        T_OUT *C,
        T_SCALE alpha,
        T_SCALE beta,
        int lda,
        int ldb,
        int ldc) {
  try {
    cublasLtMatmulDesc_t matmulDesc = NULL;
    const size_t workspaceSize = 1024 * 1024 * 4;
    void * workspace;
    int ldatransform = blas_opts.m_orderingA == CUBLASLT_ORDER_COL ? lda : 32 * lda;
    int ldbtransform = 0;
    int ldctransform = blas_opts.m_orderingC == CUBLASLT_ORDER_COL ? ldc : 32 * ldc; 

    switch(blas_opts.m_orderingB) {
      case CUBLASLT_ORDER_COL32_2R_4R4: // for ampere  
        ldbtransform = 32 * roundoff(ldb, 32);    
        break;
      case CUBLASLT_ORDER_COL:
        ldbtransform = ldb;  
        break;
      default:
        ldbtransform = 32 * roundoff(ldb, 8);  
        break;
    }

    cublas::cuda_check_error(cudaMalloc(&workspace, workspaceSize), "cudaMalloc for workspace failed");

    cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;

   
    cublas::cublas_check_error(cublasLtMatmulDescCreate(&matmulDesc, blas_opts.compute_type, blas_opts.scale_type),
                               "create MatmulDesc failed");
   

    /*
    cublas::cublas_check_error(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, blas_opts.scale_type),
        "create MatmulDesc failed");
    */

    cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &blas_opts.transb, sizeof(blas_opts.transb)),
                               "set DESC_TRANSB failed");
    cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &blas_opts.transa, sizeof(blas_opts.transa)),
                               "set DESC_TRANSA failed");
                                   cublas::cublas_check_error(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSC, &blas_opts.transc, sizeof(blas_opts.transc)),
                               "set DESC_TRANSC failed");


    // ---------------------------------------------------------------------------------------------
    // create descriptors for transformed matrices

    cublas::cublas_check_error(cublasLtMatrixLayoutCreate(&AtransformDesc, blas_opts.input_type, 
           blas_opts.transa == CUBLAS_OP_N ? blas_opts.m : blas_opts.k, blas_opts.transa == CUBLAS_OP_N ? blas_opts.k : blas_opts.m, ldatransform),
                               "create MatrixLayout for AtransformDesc failed");
    cublas::cublas_check_error(cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &blas_opts.m_orderingA, sizeof(blas_opts.m_orderingA)),
        "set LAYOUT_ORDER for AtransformDesc failed");
    cublas::cublas_check_error(cublasLtMatrixLayoutCreate(&BtransformDesc, blas_opts.input_type, 
      blas_opts.transb == CUBLAS_OP_N ? blas_opts.k : blas_opts.n, blas_opts.transb == CUBLAS_OP_N ? blas_opts.n : blas_opts.k, ldbtransform),
                               "create MatrixLayout for BtransformDesc failed");
    cublas::cublas_check_error(cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &blas_opts.m_orderingB, sizeof(blas_opts.m_orderingB)),
        "set LAYOUT_ORDER for BtransformDesc failed");

    cublas::cublas_check_error(cublasLtMatrixLayoutCreate(&CtransformDesc, blas_opts.output_type, blas_opts.m, blas_opts.n, ldctransform),
                               "create MatrixLayout for CtransformDesc failed");
    cublas::cublas_check_error(cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &blas_opts.m_orderingC, sizeof(blas_opts.m_orderingC)),
        "set LAYOUT_ORDER for CtransformDesc failed");
    // ---------------------------------------------------------------------------------------------
    // computation
    char ta = operation_to_char(blas_opts.transa);
    char tb = operation_to_char(blas_opts.transb);

    printf ("#### args: ta=%c tb=%c m=%d n=%d k=%d", ta, tb, blas_opts.m, blas_opts.n, blas_opts.k);
    //printCuType( " alpha =", alpha);
    //printCuType( " beta=", beta);
    printf("\n");
    printf("#### args: lda=%d ldb=%d ldc=%d loop=%d\n", ldatransform, ldbtransform, ldctransform, blas_opts.timing_loop);   
    printf("#### input_type %d output_type %d scale_type %d math_type %d compute_type %d\n",
        blas_opts.input_type, blas_opts.output_type, blas_opts.scale_type, blas_opts.math_type, blas_opts.compute_type);

    using namespace std::chrono;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (int i = 0; i < blas_opts.timing_loop; ++i) {
      cublas::cublas_check_error(cublasLtMatmul(ltHandle,
                                                matmulDesc,
                                                &alpha,
                                                A,
                                                AtransformDesc,
                                                B,
                                                BtransformDesc,
                                                &beta,
                                                C,
                                                CtransformDesc,
                                                C,
                                                CtransformDesc,
                                                NULL,
                                                workspace,
                                                workspaceSize,
                                                0),"cublasLtMatmul failed");    
    }

    cublas::cuda_check_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end - start);
    double flopsCoef = 2.0;

    if ((blas_opts.math_type == CUDA_C_32F) || (blas_opts.math_type == CUDA_C_64F)) {
      flopsCoef = 8.0;
    }

    double TheoreticalFlops = flopsCoef * (double)blas_opts.m * (double)blas_opts.n * (double)blas_opts.k;    
    double  TheoreticalBandwidth =
                sizeof(T_IN) * ((double)blas_opts.m * (double)blas_opts.k + (double)blas_opts.k * (double)blas_opts.n) +
                sizeof(T_OUT) * (double)blas_opts.m * (double)blas_opts.n;    
    // if (blas_opts.timing_only) {
      // fprintf (stdout, "!!!! GPU timing only. CPU reference not run.\n");
      double cudaGflops = blas_opts.timing_loop * (1e-9*TheoreticalFlops)/(time_span.count());
      double cudaBandwidth = blas_opts.timing_loop *(1e-9*TheoreticalBandwidth)/(time_span.count());
      cublasPrintPerf(false, time_span.count(), cudaGflops);//, cudaBandwidth );    
      // printGemmSOL<T_MATH>(0/*mathMode*/, time_span.count(), blas_opts.timing_loop, blas_opts.m, blas_opts.n, blas_opts.k, (int)(blas_opts.algo));
    // }

    // descriptors are no longer needed as all GPU work was already enqueued
    if (CtransformDesc) cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(CtransformDesc), "destory CtransformDesc failed");
    if (BtransformDesc) cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(BtransformDesc), "destory BtransformDesc failed");
    if (AtransformDesc) cublas::cublas_check_error(cublasLtMatrixLayoutDestroy(AtransformDesc), "destory AtransformDesc failed");
    if (matmulDesc) cublas::cublas_check_error(cublasLtMatmulDescDestroy(matmulDesc), "destroy matmulDesc failed");
    if (workspace) cublas::cuda_check_error(cudaFree(workspace), "free workspace failed");    
  } catch (cublas::cuda_exception &e) {
    cout << e << endl;  
    return 1;
  } catch (cublas::cublas_exception &e) {
    cout << e << endl;  
    return 1;
  } catch (const std::exception & e){
    cout << e.what() << endl;  
    return 1;  
  }
 
  return 0;
}

template <typename T_IN, typename T_OUT, typename T_MATH, typename T_SCALE>
static void
test_engine(const BlasOpts& blas_opts) {
  /* printf("testing cublasLt\n"); */
  try {
    T_IN *d_A = nullptr;
    T_IN *d_B = nullptr;      
    T_OUT *d_C = nullptr;
    T_SCALE alpha = cuGet<T_SCALE>(blas_opts.alpha);
    T_SCALE beta = cuGet<T_SCALE>(blas_opts.beta);
    int matrixM = 0, matrixN = 0, matrixK = 0;
    int rowsA = 0, rowsB = 0, rowsC = 0;
    int colsA = 0, colsB = 0, colsC = 0;
    size_t matrixSizeA = 0, matrixSizeB = 0, matrixSizeC = 0;    

    matrixM = blas_opts.m;
    matrixN = blas_opts.n;
    matrixK = blas_opts.k;
    
    if (blas_opts.lda)  {
      if ((blas_opts.transa == CUBLAS_OP_N) && (blas_opts.lda < matrixM)) {
          fprintf (stdout, "lda(=%d) must be bigger than m(=%d)\n", blas_opts.lda , matrixM);
          return;
      }
      if ((blas_opts.transa != CUBLAS_OP_N) && (blas_opts.lda < matrixK)) {
          fprintf (stdout, "lda(=%d) must be bigger than k(=%d) for ta\n", blas_opts.lda , matrixK);
          return;
      }
    }
    if (blas_opts.ldb)  {
      if ((blas_opts.transb == CUBLAS_OP_N) && (blas_opts.ldb < matrixK)) {
          fprintf (stdout, "ldb(=%d) must be bigger than k(=%d)\n", blas_opts.ldb , matrixK);
          return;
      }
      if ((blas_opts.transb != CUBLAS_OP_N) && (blas_opts.ldb < matrixN)) {
          fprintf (stdout, "ldb(=%d) must be bigger than n(=%d) for tb\n", blas_opts.ldb , matrixN);
          return;
      }

    }
    if ((blas_opts.ldc) && (blas_opts.ldc < matrixM)) {
      fprintf (stdout, "ldc(=%d) must be bigger than m(=%d)\n", blas_opts.ldc , matrixM);
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

    d_A = cublas::device_memory::allocate<T_IN>(matrixSizeA);
    d_B = cublas::device_memory::allocate<T_IN>(matrixSizeB);
    d_C = cublas::device_memory::allocate<T_OUT>(matrixSizeC);
    
    //cublas::cuda_check_error(cudaMemset(d_C, 0, matrixSizeC * sizeof(h_C[0])), "cudaMemset error");
    
    cublasLtHandle_t ltHandle;
    cublas::cublas_check_error(cublasLtCreate(&ltHandle), "create cublasLt handle failed");
    

    if(lt_gemm<T_IN, T_OUT, T_MATH, T_SCALE>(ltHandle, blas_opts, d_A, d_B, d_C, alpha, beta, rowsA, rowsB, rowsC)) {
      has_error = true;  
    }
    test_ran = true;

    cublas::device_memory::free(d_A);
    cublas::device_memory::free(d_B);
    cublas::device_memory::free(d_C);
    cublas::cublas_check_error(cublasLtDestroy(ltHandle), "destroy ltHandle failed");

/*
    if (has_error) {
      printf("testing cublasLt fail\n");    
      exit(-1);
    } else {
      printf("testing cublasLt pass\n");
    }
*/

  } catch (cublas::cuda_exception &e) {
    cout << e << endl;  
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (cublas::cublas_exception &e) {
    cout << e << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (const std::exception & e){
    cout << e.what() << endl;
    printf("testing cublasLt fail\n");
    exit(-1);
  }

}

static void
test_cublasLt(BlasOpts& blas_opts) {
  try{    
    switch(blas_opts.math_type) {
      case CUDA_R_32F: //sss A,B : FP32 ->  C FP32
        if ((blas_opts.input_type == CUDA_R_32F) &&
            (blas_opts.output_type == CUDA_R_32F) &&
            (blas_opts.scale_type == CUDA_R_32F)) {
          test_engine<float, float, float, float>(blas_opts);
        } //hss A,B FP16 ->  C FP32 
        if ((blas_opts.input_type == CUDA_R_16F) &&
            (blas_opts.output_type == CUDA_R_32F) &&
            (blas_opts.scale_type == CUDA_R_32F)) {
          test_engine<__half, float, float, float>(blas_opts);
        } // hsh A,B FP16 ->  C FP16
        if ((blas_opts.input_type == CUDA_R_16F) &&
            (blas_opts.output_type == CUDA_R_16F) &&
            (blas_opts.scale_type == CUDA_R_32F)) {
          test_engine<__half, __half, float, float>(blas_opts);
        } 
        break;
      case CUDA_C_32F: //ccc
        if ((blas_opts.input_type == CUDA_C_32F) &&
            (blas_opts.output_type == CUDA_C_32F) &&
            (blas_opts.scale_type == CUDA_C_32F)) {
          test_engine<cuComplex, cuComplex, cuComplex, cuComplex>(blas_opts);
        } 
        break; 
      case CUDA_R_64F: //ddd A,B : FP64 ->  C FP64
        if ((blas_opts.input_type == CUDA_R_64F) &&
            (blas_opts.output_type == CUDA_R_64F) &&
            (blas_opts.scale_type == CUDA_R_64F)) {
          test_engine<double, double, double, double>(blas_opts);
        } 
        break;
      case CUDA_C_64F: // zzz 
        if ((blas_opts.input_type == CUDA_C_64F) &&
            (blas_opts.output_type == CUDA_C_64F) &&
            (blas_opts.scale_type == CUDA_C_64F)) {
          test_engine<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex>(blas_opts);
        }  
        break;
      case CUDA_R_16F: // hhh   
        if ((blas_opts.input_type == CUDA_R_16F) &&
            (blas_opts.output_type == CUDA_R_16F) &&
            (blas_opts.scale_type == CUDA_R_16F)) {
          test_engine<__half, __half, __half,__half>(blas_opts);
        } 
        break;
      case CUDA_R_32I: {//bisb_imma
          int device_version = 0;
          cublas::cuda_check_error(get_device_version(device_version), "get device version failed");          
          if (device_version < 750) {
            printf("not supported for the imma options\n");
	        test_ran = false;
            return;
          }
          blas_opts.m_orderingA = CUBLASLT_ORDER_COL32;
          blas_opts.m_orderingB = device_version >= 800 ? CUBLASLT_ORDER_COL32_2R_4R4 : CUBLASLT_ORDER_COL4_4R2_8C;
          blas_opts.m_orderingC = CUBLASLT_ORDER_COL32;
          blas_opts.transa = CUBLAS_OP_N;
          blas_opts.transb = CUBLAS_OP_T; 
          if ((blas_opts.input_type == CUDA_R_8I) &&
              (blas_opts.output_type == CUDA_R_8I) &&
              (blas_opts.scale_type == CUDA_R_32F)) {
              
            test_engine<int8_t, int8_t, int, float>(blas_opts);
          } //bii_imma
          if ((blas_opts.input_type == CUDA_R_8I) &&
              (blas_opts.output_type == CUDA_R_32I) &&
              (blas_opts.scale_type == CUDA_R_32I)) {
            test_engine<int8_t, int, int, int>(blas_opts);
          } 
        }
        break;
      default:
        printf("mode not supported\n");
        break;
    }    
  } catch (cublas::cuda_exception &e) {
    cout << e << endl;  
    printf("testing cublasLt fail\n");
    exit(-1);
  } catch (const std::exception & e){
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
  WGST wgst;

  sem_init(&wd, 0, 0);
  sem_init(&go, 0, 0);
  sem_init(&done, 0, 0);
  void(*watchdog(void*));

  if (pthread_attr_init(&attr)) {
    perror("pthread_attr_init - watchdog");
    exit(-1);
  }

  if (ret = pthread_create(&wd_thread, &attr, watchdog, (void*)NULL) != 0) {
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
  BlasOpts blas_opts;
  parse_args(command_line, blas_opts);
  reset_blas_opts(command_line, blas_opts);

  /* GPU detection and test initilization*/
      int dev;
      size_t gpumem = 0LL;
      cudaDeviceProp devprops[MAX_NUM_GPUS] {};
      for (dev = 0; dev < deviceCount; dev++) {
          CHECK(cudaSetDevice(dev));
          CHECK(cudaGetDeviceProperties(&devprops[dev], dev));
          printf("Device %d: \"%s\"\n", dev, devprops[dev].name);
          if (dev == 0)
              gpumem = devprops[dev].totalGlobalMem;
          else {
              if (gpumem != devprops[dev - 1].totalGlobalMem) {
                  printf("Detected different GPU memory sizes\n");
                  printf("gpumem: %lld, GPU %d %lld\n", gpumem, (dev - 1), devprops[dev - 1].totalGlobalMem);
                  printf("EXITING...\n");
                  exit(0);
              }
          }
      }

      /* Initilize tests based on type of GPU; currently A100 or T4 based
      ** default to A100 unless T4
      */
      string gpu_name(devprops[0].name);
      if (gpu_name == "Tesla T4") {
          cout << "Initilizing T4 based test suite" << endl;
          wgst = WGST(WGST::T4);
      }
      else {
          cout << "Initilizing A100 based test suite" << endl;
          wgst = WGST(WGST::A100);
      }
      int memgb = 0;
      if (gpumem >= (39 * 1e9))
          memgb = 40;
      else if (gpumem >= (31 * 1e9))
          memgb = 32;
      else if (gpumem >= (15 * 1e9))
          memgb = 16;
      else {
          printf("Unexpected GPU memory size: %lld\n", gpumem);
          printf("EXITING\n");
          exit(0);
      }

      printf("GPU Memory: %lld, memgb: %d\n", gpumem, memgb);
      printf("\n\n");


  for (dev = 0; dev < deviceCount; dev++) {
	CHECK(cudaSetDevice(dev));
	printf("******** Running Stress Tests on Device: %d, Name: %s *********\n",dev,devprops[dev].name);

    // wgst.dump_test_args(0);

	for (int t_num = 0; t_num  < NUM_TESTS; t_num++) {

       /* Abort if watchdog has died */
        if (watchdog_bailed) {
            printf("WATCHDOG Thread exited...\n");
            printf("GPUstress terminating\n");
            exit(-1);
        }
        reset_blas_opts(command_line, blas_opts);
        /* Debug
        wgst.dump_test_args(tix);
        hello_world(blas_opts, wgst.stress_tests[0].P_arg);
        */

        /* Parse command line optioms */
        bool p_parse = parse_in_math_scale_out_type(blas_opts, wgst.stress_tests[t_num].P_arg);
        // cout << "DEBUG:" << "after parse" << endl;
		if (!p_parse) {
			printf("p_parse failed\n");
			exit(-1);
		}
        // cout << "DEBUG:" << "set opts" << endl;
		if (memgb == 16) {
		    blas_opts.m = wgst.stress_tests[t_num].m_arg;
		    blas_opts.n = wgst.stress_tests[t_num].n_arg;
		    blas_opts.k = wgst.stress_tests[t_num].k_arg;
            blas_opts.m_opt = true;
            blas_opts.n_opt = true;
            blas_opts.k_opt = true;
		} else {
		    // For right now, use the same values for 32GB and 40GB
		    blas_opts.m = (wgst.stress_tests[t_num].m_arg * 2);
		    blas_opts.n = (wgst.stress_tests[t_num].n_arg * 2);
		    blas_opts.k = (wgst.stress_tests[t_num].k_arg * 2);
            blas_opts.m_opt = true;
            blas_opts.n_opt = true;
            blas_opts.k_opt = true;
		}
        if (wgst.stress_tests[t_num].ta_arg == 1)
            blas_opts.transa_opt = true;
        else {
            blas_opts.transa_opt = false;
            blas_opts.transa = (cublasOperation_t)0;
        }
        if (wgst.stress_tests[t_num].tb_arg == 1)
            blas_opts.transb_opt = true;
        else {
            blas_opts.transb_opt = false;
            blas_opts.transb = (cublasOperation_t)0;
        }
        blas_opts.beta = 0.0f;
        blas_opts.beta_opt = true;
        
        printf("***** STARTING TEST %d: %s On Device %d %s\n", t_num, wgst.stress_tests[t_num].test_name, dev, devprops[dev].name);
        tstate[t_num].test_name = wgst.stress_tests[t_num].test_name;
        tstate[t_num].test_state = 1;
        tstate[t_num].start_time = time(NULL);
        // cout << "DEBUG:" << "signal wd" << endl;
        
        /* Signal watchdog test started */
        sem_post(&wd);
        // cout << "DEBUG:" << "start test" << endl;

        /* Run the test */
        test_cublasLt(blas_opts);
		printf("***** TEST %s On Device %d %s\n", wgst.stress_tests[t_num].test_name, dev, devprops[dev].name);
		if (!test_ran) 
			printf("***** TEST DID NOT EXECUTE *****\n\n");
		else {
			if (has_error == true)
				printf("***** TEST FAILED ****\n\n");
			else
				printf("***** TEST PASSED ****\n\n");
		}
        tstate[t_num].end_time = time(NULL);
        tstate[t_num].test_state = 0;
	    printf("TEST TIME: %d seconds\n",(int)(tstate[t_num].end_time - tstate[t_num].start_time));
        if (t_num == NUM_TESTS)
            tests_done = 1;

            /* Signal watchdog test finished*/
            sem_post(&done);

        /* wait for watchdog to signal next next test */
        sem_wait(&go);
	}
  }
  
  return ret;
}
