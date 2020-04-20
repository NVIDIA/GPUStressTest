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
  cudaDataType_t input_type;
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
  cublasLtOrder_t m_orderingA;
  bool m_orderingA_opt;
  cublasLtOrder_t m_orderingB;
  bool m_orderingB_opt;
  cublasLtOrder_t m_orderingC;
  bool m_orderingC_opt;
};

template <typename T_MATH> void printGemmSOL(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);

/* Common routines to print results in a uniform way (easier to parse) */
void cublasPrintPerf( bool csv,     double cudaTime, double cudaGflops, double cudaBandwithGb = -1,
                      const char *cpuLib = NULL, double cpuTime = -1,  double cpuGflops = -1 , double cpuBandwithGb= -1);

cudaError_t get_device_version(int &device_version);
