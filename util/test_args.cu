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
#include "test_args.h"
#include "type_convert.h"

/*------------------------------------------------------------*/
/* Explicit instanciation of printCuType */
template <>
void printCuType(const char *str, int A) {
  fprintf(stdout, "%s (0x%04x, %d)", str, A, A);
}
template <>
void printCuType(const char *str, float A) {
  fprintf(stdout, "%s (0x%08x, %g)", str, floatAsUInt(A), A);
}
template <>
void printCuType(const char *str, __half A) {
  __half_raw hr_A = *(reinterpret_cast<__half_raw *>(&A));

  float Af = cuGet<float>(A);
  fprintf(stdout, "%s (0x%04x, %g)", str, hr_A.x, Af);
}
template <>
void printCuType(const char *str, int8_t A) {
  fprintf(stdout, "%s (0x%04x, %d)", str, A, A);
}
template <>
void printCuType(const char *str, unsigned char A) {
  fprintf(stdout, "%s (0x%04x, %d)", str, A, A);
}
template <>
void printCuType(const char *str, double A) {
  fprintf(stdout, "%s (0x%016llx, %g)", str, doubleAsULL(A), A);
}
template <>
void printCuType(const char *str, cuComplex A) {
  fprintf(stdout, "%s (0x%08x %g), (0x%08x %g)", str, floatAsUInt(cuCrealf(A)),
          cuCrealf(A), floatAsUInt(cuCimagf(A)), cuCimagf(A));
}
template <>
void printCuType(const char *str, cuDoubleComplex A) {
  fprintf(stdout, "%s (0x%016llx %g), (0x%016llx %g)", str,
          doubleAsULL(cuCreal(A)), cuCreal(A), doubleAsULL(cuCimag(A)),
          cuCimag(A));
}

cublasComputeType_t cudaDataType2computeType(cudaDataType_t type,
                                                           bool pedantic) {
  switch (type) {
    case CUDA_R_32F:
    case CUDA_C_32F:
      return pedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    case CUDA_R_16F:
      return pedantic ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F;
    case CUDA_R_64F:
    case CUDA_C_64F:
      return pedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
    case CUDA_R_32I:
      return pedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
    default:
      return cublasComputeType_t(-1);
  }
}

#define EPILOGUE_MAPPING(key, value) \
  if (epilogue == key) {             \
    blas_opts.m_epilogue = value;    \
    return true;                     \
  }

static bool parse_epilogue(BlasOpts &blas_opts, const string &epilogue) {
  EPILOGUE_MAPPING("Default", CUBLASLT_EPILOGUE_DEFAULT)
  EPILOGUE_MAPPING("Bias", CUBLASLT_EPILOGUE_BIAS)
  EPILOGUE_MAPPING("Gelu", CUBLASLT_EPILOGUE_GELU)
  EPILOGUE_MAPPING("ReLu", CUBLASLT_EPILOGUE_RELU)
  EPILOGUE_MAPPING("ReLuBias", CUBLASLT_EPILOGUE_RELU_BIAS)
  EPILOGUE_MAPPING("GeluBias", CUBLASLT_EPILOGUE_GELU_BIAS)
  return false;
}
#undef EPILOGUE_MAPPING

#define IN_OUT_MATH_SCALE_TYPE_MAPPING(key, T_IN_A, T_IN_B, T_IN_C, T_OUT, \
                                       T_MATH, T_SCALE, T_COMPUTE)         \
  if (in_math_scale_out_type == key) {                                     \
    blas_opts.case_name = key;                                             \
    blas_opts.input_type_a = T_IN_A;                                       \
    blas_opts.input_type_b = T_IN_B;                                       \
    blas_opts.input_type_c = T_IN_C;                                       \
    blas_opts.output_type = T_OUT;                                         \
    blas_opts.math_type = T_MATH;                                          \
    blas_opts.scale_type = T_SCALE;                                        \
    blas_opts.compute_type = T_COMPUTE;                                    \
    return true;                                                           \
  }

static bool parse_in_math_scale_out_type(BlasOpts &blas_opts,
                                         const string &in_math_scale_out_type) {
  IN_OUT_MATH_SCALE_TYPE_MAPPING("sss", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                 cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("sss_fast_tf32", CUDA_R_32F, CUDA_R_32F,
                                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                 CUBLAS_COMPUTE_32F_FAST_TF32)
  IN_OUT_MATH_SCALE_TYPE_MAPPING("ccc", CUDA_C_32F, CUDA_C_32F, CUDA_C_32F,
                                 CUDA_C_32F, CUDA_C_32F, CUDA_C_32F,
                                 cudaDataType2computeType(CUDA_C_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("hhh", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                 CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                 cudaDataType2computeType(CUDA_R_16F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("ddd", CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
                                 CUDA_R_64F, CUDA_R_64F, CUDA_R_64F,
                                 cudaDataType2computeType(CUDA_R_64F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("zzz", CUDA_C_64F, CUDA_C_64F, CUDA_C_64F,
                                 CUDA_C_64F, CUDA_C_64F, CUDA_C_64F,
                                 cudaDataType2computeType(CUDA_C_64F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("hsh", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                 CUDA_R_16F, CUDA_R_32F, CUDA_R_32F,
                                 cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("hss", CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
                                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                 cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("bisb_imma", CUDA_R_8I, CUDA_R_8I, CUDA_R_8I,
                                 CUDA_R_8I, CUDA_R_32I, CUDA_R_32F,
                                 CUBLAS_COMPUTE_32I)
  IN_OUT_MATH_SCALE_TYPE_MAPPING("bii_imma", CUDA_R_8I, CUDA_R_8I, CUDA_R_32I,
                                 CUDA_R_32I, CUDA_R_32I, CUDA_R_32I,
                                 CUBLAS_COMPUTE_32I)
  IN_OUT_MATH_SCALE_TYPE_MAPPING(
      "qqssq", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_8F_E4M3,
      CUDA_R_32F, CUDA_R_32F, cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING(
      "mxqqhsq", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_8F_E4M3,
      CUDA_R_32F, CUDA_R_32F, cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING(
      "nvoohso", CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16F, CUDA_R_4F_E2M1,
      CUDA_R_32F, CUDA_R_32F, cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("tss", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                                 CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
                                 cudaDataType2computeType(CUDA_R_32F, false))
  IN_OUT_MATH_SCALE_TYPE_MAPPING("tst", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
                                 CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F,
                                 cudaDataType2computeType(CUDA_R_32F, false))

  return false;
}

#undef IN_OUT_MATH_SCALE_TYPE_MAPPING

char operation_to_char(cublasOperation_t op) {
  switch (op) {
    case CUBLAS_OP_T:
      return 'T';
    case CUBLAS_OP_C:
      return 'C';
    case CUBLAS_OP_N:
      return 'N';
    default:
      return 'T';
  }
}

static bool parse_operation(int in_op, cublasOperation_t &out_op) {
  switch (in_op) {
    case 0:
      out_op = CUBLAS_OP_N;
      return true;
    case 1:
      out_op = CUBLAS_OP_T;
      return true;
    case 2:
      out_op = CUBLAS_OP_C;
      return true;
    default:
      return false;
  }
}

static void usage(void) {
  printf(" <options>\n");
  printf("-h          : display this help\n");
  printf("-help       : display this help\n");
  printf("Test-specific options :\n");
  printf(
      "-P={sss,sss_fast_tf32,ccc,hhh,ddd,zzz,hsh,hss,bisb_imma,bii_imma,qqssq,"
      "tss,tst}"
      "\n");
  printf(
      "    sss: input type: float, math type: float, scale type: float, output "
      "type: float\n");
  printf(
      "    sss_fast_tf32: input type: float, math type: float, scale type: "
      "float, output type: float\n");
  printf(
      "    ccc: input type: complex, math type: complex, scale type: complex, "
      "output type: complex\n");
  printf(
      "    hhh: input type: half, math type: half, scale type: half, output "
      "type: half\n");
  printf(
      "    ddd: input type: double, math type: double, scale type: double, "
      "output type: double\n");
  printf(
      "    zzz: input type: double complex, math type: double complex, scale "
      "type: double complex, output type: double complex\n");
  printf(
      "    hsh: input type: half, math type: float, output type: half , scale "
      "type: float\n");
  printf(
      "    hss: input type: half, math type: float, output type: float, scale "
      "type: float\n");
  printf(
      "    bisb_imma: input type: int8, math type: int32, scale type: float, "
      "output type: int8\n");
  printf(
      "    bii_imma: input type: int8, math type: int32, scale type: int32, "
      "output type: int32\n");
  printf(
      "    qqssq: input type a: fp8_e4m3, input type b: fp8_e4m3, input type "
      "c: bfloat16, math type: float, scale type: float, output type: "
      "fp8_e4m3\n");
  printf(
      "  mxqqhsq: input type a: fp8_e4m3, input type b: fp8_e4m3, input type "
      "c: half, math type: float, scale type: float, block scale type: fp8_e8m0, block size: 32, output type: "
      "fp8_e4m3\n");
  printf(
      "  nvoohso: input type a: fp4_e2m1, input type b: fp4_e2m1, input type "
      "c: half, math type: float, scale type: float, block scale type: fp8_e4m3, block size: 16, output type: "
      "fp4_e2m1\n");
  printf(
      "    tss: input type: bfloat16, math type: float, output type: float , "
      "scale "
      "type: float\n");
  printf(
      "    tst: input type: bfloat16, math type: float, output type: bfloat16 "
      ", scale "
      "type: float\n");
  printf("-m=<int>  : number of rows of A and C\n");
  printf("-n=<int>  : number of columns of B and C\n");
  printf("-k=<int>  : number of columns of A and rows of B\n");
  printf("-A=<float>  : value of alpha\n");
  printf("-B=<float>  : value of beta\n");
  printf(
      "-T=<int> : run N times back to back  , good for power consumption, no "
      "results checking\n");
  printf("-W=<int> : run N times for warmup (default : 100)\n");
  printf("-lda=<int> : leading dimension of A , m by default\n");
  printf("-ldb=<int> : leading dimension of B , k by default\n");
  printf("-ldc=<int> : leading dimension of C , m by default\n");
  printf("-ta= op(A) {0=no transpose, 1=transpose, 2=hermitian}\n");
  printf("-tb= op(B) {0=no transpose, 1=transpose, 2=hermitian}\n");
  printf(
      "-p=<0|z|r|t|P> : 0 or z:fill all matrices with zero, r: fill with normal distribution, "
      "t: uniform with random signs, otherwise fill with pseudorandom distribution\n");
  printf("-sd=<float> : standard deviation for normal distribution filling mode\n");
  printf("-m_outOfPlace=<int> : out of place (C != D), 0: disable, 1:enable\n");
  printf("-m_epilogue={Default,Bias,Gelu,ReLu,ReLuBias,GeluBias}\n");
  printf("-z<a|b|c|d>=<0|1> : zero-copy , A,B,C or D is pinned on the Host, 0: disable, 1:enable\n");
  printf("-s : shows the CUDA configuration of the machines\n");
  printf("-C=<0|1> : check available memory before running a test, 0: disable, 1:enable\n");
  printf("-strided_batch=<0|1> : run strided batched GEMM API, 0: disable, 1:enable\n");
  printf("-N=<int> : number of matrices to run test\n");
  printf("-batch_stride<A,B,C,D>=<int64_t> : offset between matrices in strided batched GEMM\n");
}

void parse_args(CommandLine &command_line, BlasOpts &blas_opts) {
  memset((void *)&blas_opts, 0, sizeof(BlasOpts));
  blas_opts.timing_only = false;
  blas_opts.transa = DEFAULT_TRANS_OP_N;
  blas_opts.transb = DEFAULT_TRANS_OP_N;
  blas_opts.input_type_a = DEFAULT_DATA_TYPE_FP32;
  blas_opts.input_type_b = DEFAULT_DATA_TYPE_FP32;
  blas_opts.input_type_c = DEFAULT_DATA_TYPE_FP32;
  blas_opts.output_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.compute_type = DEFAULT_COMPUTE_TYPE_32F;
  blas_opts.scale_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.math_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.algo = DEFAULT_ALGO_GEMM_DEFAULT;
  blas_opts.m = DEFAULT_1024;
  blas_opts.n = DEFAULT_1024;
  blas_opts.k = DEFAULT_1024;
  blas_opts.timing_loop = 1;
  blas_opts.warmup_loops = 100;
  blas_opts.m_orderingA = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingB = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingC = CUBLASLT_ORDER_COL;
  blas_opts.alpha = 1.0f;
  blas_opts.beta = 1.0f;
  blas_opts.fillingPattern = 'P';
  blas_opts.filling_sd = 1.0;
  blas_opts.filling_mean = 0.0;
  blas_opts.m_outOfPlace = false;
  blas_opts.m_epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  blas_opts.quick_autotuning = false;
  blas_opts.enableZeroCopy = false;
  blas_opts.check = false;
  blas_opts.N = 1;
  blas_opts.useBatch = false;

  if (command_line.check_cmd_line_flag("m")) {
    command_line.get_cmd_line_argument("m", blas_opts.m);
    blas_opts.m_opt = true;
  }
  if (command_line.check_cmd_line_flag("n")) {
    command_line.get_cmd_line_argument("n", blas_opts.n);
    blas_opts.n_opt = true;
  }
  if (command_line.check_cmd_line_flag("k")) {
    command_line.get_cmd_line_argument("k", blas_opts.k);
    blas_opts.k_opt = true;
  }
  if (command_line.check_cmd_line_flag("lda")) {
    command_line.get_cmd_line_argument("lda", blas_opts.lda);
  }
  if (command_line.check_cmd_line_flag("ldb")) {
    command_line.get_cmd_line_argument("ldb", blas_opts.ldb);
  }
  if (command_line.check_cmd_line_flag("ldc")) {
    command_line.get_cmd_line_argument("ldc", blas_opts.ldc);
  }
  if (command_line.check_cmd_line_flag("A")) {
    command_line.get_cmd_line_argument("A", blas_opts.alpha);
    blas_opts.alpha_opt = true;
  }
  if (command_line.check_cmd_line_flag("B")) {
    command_line.get_cmd_line_argument("B", blas_opts.beta);
    blas_opts.beta_opt = true;
  }
  if (command_line.check_cmd_line_flag("B")) {
    command_line.get_cmd_line_argument("B", blas_opts.beta);
    blas_opts.beta_opt = true;
  }
  if (command_line.check_cmd_line_flag("T")) {
    command_line.get_cmd_line_argument("T", blas_opts.timing_loop);
    blas_opts.timing_only = true;
  }
  if (command_line.check_cmd_line_flag("W")) {
    command_line.get_cmd_line_argument("W", blas_opts.warmup_loops);
  }
  if ((command_line.check_cmd_line_flag("h")) ||
      (command_line.check_cmd_line_flag("help"))) {
    usage();
    exit(0);
  }
  if (command_line.check_cmd_line_flag("P")) {
    string in_math_scale_out_type;
    command_line.get_cmd_line_argument("P", in_math_scale_out_type);
    if (!parse_in_math_scale_out_type(blas_opts, in_math_scale_out_type)) {
      fprintf(stdout, "Option P=%s not supported\n",
              in_math_scale_out_type.c_str());
      usage();
      exit(-1);
    }
  }
  if (command_line.check_cmd_line_flag("ta")) {
    int transa = 0;
    command_line.get_cmd_line_argument("ta", transa);
    if (!parse_operation(transa, blas_opts.transa)) {
      fprintf(stdout, "transa=%d not supported\n", transa);
      usage();
      exit(-1);
    }
    blas_opts.transa_opt = true;
  }
  if (command_line.check_cmd_line_flag("tb")) {
    int transb = 0;
    command_line.get_cmd_line_argument("tb", transb);
    if (!parse_operation(transb, blas_opts.transb)) {
      fprintf(stdout, "transb=%d not supported\n", transb);
      usage();
      exit(-1);
    }
    blas_opts.transb_opt = true;
  }
  if (command_line.check_cmd_line_flag("p")) {
    command_line.get_cmd_line_argument("p", blas_opts.fillingPattern);
  }
  if (command_line.check_cmd_line_flag("sd")) {
    command_line.get_cmd_line_argument("sd", blas_opts.filling_sd);
  }
  if (command_line.check_cmd_line_flag("m_outOfPlace")) {
    int m_outOfPlace = 0;
    command_line.get_cmd_line_argument("m_outOfPlace", m_outOfPlace);
    if (m_outOfPlace) {
      blas_opts.m_outOfPlace = true;
    }
  }
  if (command_line.check_cmd_line_flag("m_epilogue")) {
    string epilogue;
    command_line.get_cmd_line_argument("m_epilogue", epilogue);
    if (!parse_epilogue(blas_opts, epilogue)) {
      fprintf(stdout, "Option epilogue=%s not supported\n", epilogue.c_str());
      usage();
      exit(-1);
    }
  }
  if (command_line.check_cmd_line_flag("tune")) {
    int tune = 0;
    command_line.get_cmd_line_argument("tune", tune);
    if (tune != 0) {
      blas_opts.quick_autotuning = true;
    }
  }
  if (command_line.check_cmd_line_flag("za")) {
    command_line.get_cmd_line_argument("za", blas_opts.zeroCopy[0]);
    blas_opts.enableZeroCopy = true;
  }
  if (command_line.check_cmd_line_flag("zb")) {
    command_line.get_cmd_line_argument("zb", blas_opts.zeroCopy[1]);
    blas_opts.enableZeroCopy = true;
  }
  if (command_line.check_cmd_line_flag("zc")) {
    command_line.get_cmd_line_argument("zc", blas_opts.zeroCopy[2]);
    blas_opts.enableZeroCopy = true;
  }
  if (command_line.check_cmd_line_flag("zd")) {
    command_line.get_cmd_line_argument("zd", blas_opts.zeroCopy[3]);
    blas_opts.enableZeroCopy = true;
  }
  if (command_line.check_cmd_line_flag("s")) {
    exit(showDevices(0));
  }
  if (command_line.check_cmd_line_flag("C")) {
    int check = 0;
    command_line.get_cmd_line_argument("C", check);
    if (check != 0) {
      blas_opts.check = true;
    }
  }
  if (command_line.check_cmd_line_flag("N")) {
    int N = 0;
    command_line.get_cmd_line_argument("N", N);
    if (1 != N) {
      if (command_line.check_cmd_line_flag("strided_batch")) {
        int strided_batch = 0;
        command_line.get_cmd_line_argument("strided_batch", strided_batch);
        if (strided_batch != 0) {
          blas_opts.N = N;
        } else {
          printf("^^^^ Stride batched is disable, N should be 1\n");
        }
      } else {
        printf("^^^^ Stride batched is disable, N should be 1\n");
      }
    }
  }
  if (command_line.check_cmd_line_flag("strided_batch")) {
    int strided_batch = 0;
    command_line.get_cmd_line_argument("strided_batch", strided_batch);
    if (strided_batch != 0) {
      blas_opts.useBatch = true;
    }
  }
  if (command_line.check_cmd_line_flag("batch_strideA")) {
    command_line.get_cmd_line_argument("batch_strideA", blas_opts.batch_stride[0]);
    blas_opts.batch_strideOpt[0] = true;
  }
  if (command_line.check_cmd_line_flag("batch_strideB")) {
    command_line.get_cmd_line_argument("batch_strideB", blas_opts.batch_stride[1]);
    blas_opts.batch_strideOpt[1] = true;
  }
  if (command_line.check_cmd_line_flag("batch_strideC")) {
    command_line.get_cmd_line_argument("batch_strideC", blas_opts.batch_stride[2]);
    blas_opts.batch_strideOpt[2] = true;
  }
  if (command_line.check_cmd_line_flag("batch_strideD")) {
    command_line.get_cmd_line_argument("batch_strideD", blas_opts.batch_stride[3]);
    blas_opts.batch_strideOpt[3] = true;
  }
  if (!command_line.all_flags_checked()) {
    exit(-1);
  }
}

void reset_blas_opts(CommandLine& command_line, BlasOpts &blas_opts)
{
  memset ((void *)&blas_opts, 0, sizeof(BlasOpts));
  parse_args(command_line, blas_opts);
  memset((void *)&blas_opts, 0, sizeof(BlasOpts));
  blas_opts.timing_only = false;
  blas_opts.transa = DEFAULT_TRANS_OP_N;
  blas_opts.transb = DEFAULT_TRANS_OP_N;
  blas_opts.input_type_a = DEFAULT_DATA_TYPE_FP32;
  blas_opts.input_type_b = DEFAULT_DATA_TYPE_FP32;
  blas_opts.input_type_c = DEFAULT_DATA_TYPE_FP32;
  blas_opts.output_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.compute_type = DEFAULT_COMPUTE_TYPE_32F;
  blas_opts.scale_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.math_type = DEFAULT_DATA_TYPE_FP32;
  blas_opts.algo = DEFAULT_ALGO_GEMM_DEFAULT;
  blas_opts.m = DEFAULT_1024;
  blas_opts.n = DEFAULT_1024;
  blas_opts.k = DEFAULT_1024;
  if (command_line.check_cmd_line_flag("T")) {
     command_line.get_cmd_line_argument("T", blas_opts.timing_loop);
 }
 else {
     blas_opts.timing_loop = 10;
 }
  blas_opts.timing_loop = 1;
  blas_opts.m_orderingA = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingB = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingC = CUBLASLT_ORDER_COL;
  blas_opts.alpha = 1.0f;
  blas_opts.beta = 1.0f;
  blas_opts.m_outOfPlace = false;
  blas_opts.m_epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  blas_opts.quick_autotuning = false;
}

