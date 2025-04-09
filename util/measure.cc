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

/*----------------------------------------------------------------------------*/
/* Common routines to print results in a uniform way (easier to parse) */
/* void cublasPrintPerf( bool csv,     double cudaTime, double cudaGflops, double cudaBandwidthGb,
                      const char *cpuLib = NULL, double cpuTime = -1,  double cpuGflops = -1 , double cpuBandwidthGb= -1);
*/
void cublasPrintPerf( bool csv,     double cudaTime, double cudaGflops, double cudaBandwidthGb,
                      const char *cpuLib, double cpuTime,  double cpuGflops, double cpuBandwidthGb)
{
  if (csv) {
  /* CSV Format is as follows :
    ####CSV cudaTime, cudaGflops, cudaBandwidth, cpuTime, cpuGflops, cpuBandwidth, speedup
    if a field is not significant, only a comma is printed
  */

    printf("^^^^CSV   %g, ", cudaTime);
    if (cudaGflops > 0)    {
      printf( "%.3f, ",      cudaGflops );
    }
    else {
      printf( " , ");
    }

    if (cudaBandwidthGb > 0)    {
      printf( "%.3f, ",      cudaBandwidthGb );
    }
    else {
      printf( ", ");
    }
    if (cpuLib) {
      printf("%g, ",  cpuTime);
      if (cpuGflops > 0)    {
        printf( "%.3f, ",      cpuGflops );
      }
      else {
        printf( ", ");
      }
      if (cpuBandwidthGb > 0)    {
        printf( "%.3f, ",      cpuBandwidthGb );
      }
      else {
        printf( ", ");
      }
      printf( "%.2f,",  cpuTime/cudaTime );
    }
    else {
      printf(" , , , ,");
    }
    printf("\n");

    /*Eris perf output is only when -csv option is on */
    // if (cudaGflops > 0) {
    //   printf("&&&& PERF Gflops %.6g Gflops\n", cudaGflops);
    // } else if (cudaBandwidthGb > 0) {
    //   printf("&&&& PERF Bandwidth %.6g GB/s\n", cudaBandwidthGb);
    // }
  }

  printf( "^^^^ CUDA : elapsed = %g sec,  ",  cudaTime );
  if (cudaGflops > 0)    printf( "Gflops = %.3f ",      cudaGflops );
  if (cudaBandwidthGb > 0) printf( "Bandwidth = %.3f ",  cudaBandwidthGb );
  printf( "\n");
  if (cpuLib) {
    printf( "^^^^%s : elapsed = %g sec, ",  cpuLib, cpuTime );
    if (cpuGflops > 0)    printf( "Gflops = %.3f ",      cpuGflops );
    if (cpuBandwidthGb > 0) printf( "Bandwidth = %.3f, ",  cpuBandwidthGb );
    printf( "Speedup %.2f\n",  cpuTime/cudaTime );
  }
}

template<typename T_MATH> char gemmType();
template<> char gemmType<float>() { return 's'; }
template<> char gemmType<double>() { return 'd'; }
template<> char gemmType<__half>() { return 'h'; }
template<> char gemmType<__nv_bfloat16>() { return 't'; }
template<> char gemmType<int>() { return 'i'; }
template<> char gemmType<int8_t>() { return 'b'; }
template<> char gemmType<cuComplex>() { return 'c'; }
template<> char gemmType<cuDoubleComplex>() { return 'z'; }




