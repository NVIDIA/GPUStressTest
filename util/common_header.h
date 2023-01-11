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
 * Utility for parsing command line arguments
 */

#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdexcept>
#include <chrono>
//#include <cublas_v2.h>
#include <cublas_api.h>
//#include <cuda_bf16.h>
#include <cuda_fp8.h>



using std::string;
using std::cout;
using std::endl;

#define imax(x,y) (((x) > (y)) ? (x) : (y))

const int DEFAULT_1024 = 2048;
const cublasOperation_t  DEFAULT_TRANS_OP_N  = CUBLAS_OP_N;
const cudaDataType_t DEFAULT_DATA_TYPE_FP32 = CUDA_R_32F;
const cublasComputeType_t DEFAULT_COMPUTE_TYPE_32F = CUBLAS_COMPUTE_32F;
const int DEFAULT_ALGO_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
const cublasLtOrder_t DEFAULT_ORDER_COL = CUBLASLT_ORDER_COL;

struct BlasOpts {
  cublasOperation_t transa;
  bool transa_opt;
  cublasOperation_t transb;
  bool transb_opt;
  cublasOperation_t transc;
  bool transc_opt;
  cudaDataType_t input_type_a;
  cudaDataType_t input_type_b;
  cudaDataType_t input_type_c;
  cudaDataType_t output_type;
  cudaDataType_t scale_type;
  cudaDataType_t math_type;
  cublasComputeType_t compute_type;
  int algo;
  bool algo_opt;
  int m;
  bool m_opt;
  int n;
  bool n_opt;
  int k;
  bool k_opt;
  int lda;
  int ldb;
  int ldc;
  int timing_loop;  // For TDP, run the GPU in a loop
  bool timing_only;  // for benchmarking - we only run the GPU version
  float alpha;
  bool alpha_opt;
  float beta;
  bool beta_opt;
  bool filling_zero;
  cublasLtOrder_t m_orderingA;
  bool m_orderingA_opt;
  cublasLtOrder_t m_orderingB;
  bool m_orderingB_opt;
  cublasLtOrder_t m_orderingC;
  bool m_orderingC_opt;
  bool m_outOfPlace;
  cublasLtEpilogue_t m_epilogue;
  bool quick_autotuning;
};

template <typename T_MATH> void printGemmSOL(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);

/* Common routines to print results in a uniform way (easier to parse) */
void cublasPrintPerf( bool csv,     double cudaTime, double cudaGflops, double cudaBandwithGb = -1,
                      const char *cpuLib = NULL, double cpuTime = -1,  double cpuGflops = -1 , double cpuBandwithGb= -1);

cudaError_t get_device_version(int &device_version);
