#include "common_header.h"
#include "type_convert.h"
#include "test_args.h" 


/*------------------------------------------------------------*/
/* Explicit instanciation of printCuType */
template <>
void printCuType(const char *str, int A) {
  fprintf(stdout, "%s (0x%04x, %d)",
          str,
		  A, A);
}
template <>
void printCuType(const char *str, float A) {
  fprintf(stdout, "%s (0x%08x, %g)",
          str,
		  floatAsUInt(A), A);
}
template <>
void printCuType(const char *str, __half A) {
  __half_raw hr_A = *(reinterpret_cast<__half_raw *>(&A) );

  float Af = cuGet<float>(A);
  fprintf(stdout, "%s (0x%04x, %g)",
          str,
		  hr_A.x, Af);
}
template <>
void printCuType(const char *str, int8_t A) {
  fprintf(stdout, "%s (0x%04x, %d)",
          str,
		  A, A);
}
template <>
void printCuType(const char *str, unsigned char A) {
  fprintf(stdout, "%s (0x%04x, %d)",
          str,
		  A, A);
}
template <>
void printCuType(const char *str, double A) {
  fprintf(stdout, "%s (0x%016llx, %g)",
          str,
	      doubleAsULL(A), A);
}
template <>
void printCuType(const char *str, cuComplex A) {
  fprintf(stdout, "%s (0x%08x %g), (0x%08x %g)",
          str,
		  floatAsUInt(cuCrealf(A)), cuCrealf(A),
		  floatAsUInt(cuCimagf(A)), cuCimagf(A));
}
template <>
void printCuType(const char *str, cuDoubleComplex A) {
  fprintf(stdout, "%s (0x%016llx %g), (0x%016llx %g)",
          str,
	      doubleAsULL(cuCreal(A)), cuCreal(A),
	      doubleAsULL(cuCimag(A)), cuCimag(A));
}

static inline cublasComputeType_t
cudaDataType2computeType(cudaDataType_t type, bool pedantic) {
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

/* Debug 
void hello_world(BlasOpts& blas_opts, const string& in_math_scale_out_type) {
    cout << "DEBUG:" << "hello_world " << in_math_scale_out_type  << endl;
}
*/

bool parse_in_math_scale_out_type(BlasOpts& blas_opts, const string &in_math_scale_out_type) {

  //cout << "DEBUG:" << "parse_in_math_scale_out_type " << in_math_scale_out_type << endl;

  if (in_math_scale_out_type == "sss") {
    blas_opts.input_type = CUDA_R_32F;
    blas_opts.output_type = CUDA_R_32F;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.math_type = CUDA_R_32F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_R_32F, false);
    return true;
  } else if (in_math_scale_out_type == "sss_fast_tf32") { 
    blas_opts.input_type = CUDA_R_32F;
    blas_opts.output_type = CUDA_R_32F;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.math_type = CUDA_R_32F;
    blas_opts.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    return true;
  } else if (in_math_scale_out_type == "ccc") { 
    blas_opts.input_type = CUDA_C_32F;
    blas_opts.output_type = CUDA_C_32F;
    blas_opts.scale_type = CUDA_C_32F;
    blas_opts.math_type = CUDA_C_32F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_C_32F, false);
    return true;
  } else if (in_math_scale_out_type == "hhh") { 
    blas_opts.input_type = CUDA_R_16F;
    blas_opts.output_type = CUDA_R_16F;
    blas_opts.scale_type = CUDA_R_16F;
    blas_opts.math_type = CUDA_R_16F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_R_16F, false); 
    return true;
  } else if (in_math_scale_out_type == "ddd") { 
    blas_opts.input_type = CUDA_R_64F;
    blas_opts.output_type = CUDA_R_64F;
    blas_opts.scale_type = CUDA_R_64F;
    blas_opts.math_type = CUDA_R_64F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_R_64F, false); 
    return true;
  } else if (in_math_scale_out_type == "zzz") { 
    blas_opts.input_type = CUDA_C_64F;
    blas_opts.output_type = CUDA_C_64F;
    blas_opts.scale_type = CUDA_C_64F;
    blas_opts.math_type = CUDA_C_64F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_C_64F, false); 
    return true;
  } else if (in_math_scale_out_type == "hsh") { 
    blas_opts.input_type = CUDA_R_16F;
    blas_opts.output_type = CUDA_R_16F;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.math_type = CUDA_R_32F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_R_32F, false);
    return true;
  } else if (in_math_scale_out_type == "hss") { 
    blas_opts.input_type = CUDA_R_16F;
    blas_opts.output_type = CUDA_R_32F;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.compute_type = cudaDataType2computeType(CUDA_R_32F, false);
    return true;
  } else if (in_math_scale_out_type == "bisb_imma") { 
    blas_opts.input_type = CUDA_R_8I;
    blas_opts.output_type = CUDA_R_8I;
    blas_opts.scale_type = CUDA_R_32F;
    blas_opts.math_type = CUDA_R_32I;
    blas_opts.compute_type = CUBLAS_COMPUTE_32I;
    return true;
  } else if (in_math_scale_out_type == "bii_imma") { 
    blas_opts.input_type = CUDA_R_8I;
    blas_opts.output_type = CUDA_R_32I;
    blas_opts.scale_type = CUDA_R_32I;
    blas_opts.math_type = CUDA_R_32I;
    blas_opts.compute_type = CUBLAS_COMPUTE_32I;
    return true;
  } else { 
    return false;  
  }
}

char
operation_to_char(cublasOperation_t op) {
  switch(op) {
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

static bool
parse_operation(int in_op, cublasOperation_t & out_op) {
  switch(in_op) {
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

static void
usage ( void ){
  printf( " <options>\n");
  printf( "-h          : display this help\n");
  printf( "-help       : display this help\n");
  printf( "Test-specific options :\n");
  printf( "-P={sss,sss_fast_tf32,ccc,hhh,ddd,zzz,hsh,hss,bisb_imma,bii_imma}\n");
  printf( "    sss: input type: float, math type: float, scale type: float, output type: float\n");
  printf( "    sss_fast_tf32: input type: float, math type: float, scale type: float, output type: float\n");
  printf( "    ccc: input type: complex, math type: complex, scale type: complex, output type: complex\n");
  printf( "    hhh: input type: half, math type: half, scale type: half, output type: half\n");
  printf( "    ddd: input type: double, math type: double, scale type: double, output type: double\n");
  printf( "    zzz: input type: double complex, math type: double complex, scale type: double complex, output type: double complex\n");
  printf( "    hsh: input type: half, math type: float, output type: half , scale type: float\n");
  printf( "    hss: input type: half, math type: float, output type: float, scale type: float\n");
  printf( "    bisb_imma: input type: int8, math type: int32, scale type: float, output type: int8\n");
  printf( "    bii_imma: input type: int8, math type: int32, scale type: int32, output type: int32\n");
  printf( "-m=<int>  : number of rows of A and C\n");
  printf( "-n=<int>  : number of columns of B and C\n");
  printf( "-k=<int>  : number of columns of A and rows of B\n");
  printf( "-A=<float>  : value of alpha\n");
  printf( "-B=<float>  : value of beta\n");
  printf( "-T=<int> : run N times back to back  , good for power consumption, no results checking\n");
  printf( "-lda=<int> : leading dimension of A , m by default\n");
  printf( "-ldb<number> : leading dimension of B , k by default\n");
  printf( "-ldc<number> : leading dimension of C , m by default\n");
  printf( "-ta= op(A) {0=no transpose, 1=transpose, 2=hermitian}\n");
  printf( "-tb= op(B) {0=no transpose, 1=transpose, 2=hermitian}\n");
}

void reset_blas_opts(CommandLine& command_line, BlasOpts &blas_opts)
{
  memset ((void *)&blas_opts, 0, sizeof(BlasOpts));  
  blas_opts.timing_only = false;
  blas_opts.transa = DEFAULT_TRANS_OP_N;
  blas_opts.transb = DEFAULT_TRANS_OP_N;
  blas_opts.transa_opt = false;
  blas_opts.transb_opt = false;
  blas_opts.input_type = DEFAULT_DATA_TYPE_FP32;
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
  blas_opts.m_orderingA = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingB = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingC = CUBLASLT_ORDER_COL;
  blas_opts.alpha = 1.0f;
  blas_opts.beta = 1.0f;
}

void
parse_args(CommandLine &command_line, BlasOpts &blas_opts) {
  memset ((void *)&blas_opts, 0, sizeof(BlasOpts));  
  blas_opts.timing_only = false;
  blas_opts.transa = DEFAULT_TRANS_OP_N;
  blas_opts.transb = DEFAULT_TRANS_OP_N;
  blas_opts.transa_opt = false;
  blas_opts.transb_opt = false;
  blas_opts.input_type = DEFAULT_DATA_TYPE_FP32;
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
  blas_opts.m_orderingA = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingB = CUBLASLT_ORDER_COL;
  blas_opts.m_orderingC = CUBLASLT_ORDER_COL;
  blas_opts.alpha = 1.0f;
  blas_opts.beta = 1.0f;

  if (command_line.check_cmd_line_flag("m")){
    command_line.get_cmd_line_argument("m", blas_opts.m);
    blas_opts.m_opt = true;
  }
  if (command_line.check_cmd_line_flag("n")){
    command_line.get_cmd_line_argument("n", blas_opts.n);
    blas_opts.n_opt = true;
  }
  if (command_line.check_cmd_line_flag("k")){
    command_line.get_cmd_line_argument("k", blas_opts.k);
    blas_opts.k_opt = true;
  }
  if (command_line.check_cmd_line_flag("lda")){
    command_line.get_cmd_line_argument("lda", blas_opts.lda);
  }
  if (command_line.check_cmd_line_flag("ldb")){
    command_line.get_cmd_line_argument("ldb", blas_opts.ldb);
  }
  if (command_line.check_cmd_line_flag("ldc")){
    command_line.get_cmd_line_argument("ldc", blas_opts.ldc);
  }
  if (command_line.check_cmd_line_flag("A")){
    command_line.get_cmd_line_argument("A", blas_opts.alpha);
    blas_opts.alpha_opt = true;
  }
  if (command_line.check_cmd_line_flag("B")){
    command_line.get_cmd_line_argument("B", blas_opts.beta);
    blas_opts.beta_opt = true;
  }
  if (command_line.check_cmd_line_flag("B")){
    command_line.get_cmd_line_argument("B", blas_opts.beta);
    blas_opts.beta_opt = true;
  }
  if (command_line.check_cmd_line_flag("T")){
    command_line.get_cmd_line_argument("T", blas_opts.timing_loop);
    blas_opts.timing_only = true;
  }
  if ((command_line.check_cmd_line_flag("h")) || (command_line.check_cmd_line_flag("help"))){
    usage();
    exit( 0 );
  }
  if (command_line.check_cmd_line_flag("P")) {
    string in_math_scale_out_type;
    command_line.get_cmd_line_argument("P", in_math_scale_out_type);
    if(!parse_in_math_scale_out_type(blas_opts, in_math_scale_out_type)) {
      fprintf(stdout, "Option P=%s not supported\n", in_math_scale_out_type.c_str());
      usage();
      exit( -1 );
    }
  }
  if (command_line.check_cmd_line_flag("ta")) {
    int transa = 0;  
    command_line.get_cmd_line_argument("ta", transa);
    if(!parse_operation(transa, blas_opts.transa)) {
      fprintf(stdout, "transa=%d not supported\n", transa);
      usage();
      exit( -1 );
    }
    blas_opts.transa_opt = true; 
  }  
  if (command_line.check_cmd_line_flag("tb")) {
    int transb = 0;  
    command_line.get_cmd_line_argument("tb", transb);
    if(!parse_operation(transb, blas_opts.transb)) {
      fprintf(stdout, "transb=%d not supported\n", transb);  
      usage();
      exit( -1 );
    }
    blas_opts.transb_opt = true; 
  }

  if(!command_line.all_flags_checked()){
    exit(-1);
  }
}


