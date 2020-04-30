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

// Coefficient to apply to the flop count where sgemm is taken as the reference
template<typename T_MATH> double coefGemmSOL(int mathMode, int major, int minor, int algorithm);
template<> double coefGemmSOL<float>(int mathMode, int major, int minor, int algorithm) {
  if(major >= 7) {
    //get mathMode or algorithm value
    if(mathMode == 1 || algorithm >= 99) {
      return 8.0;//1024.0 / 128.0;
    }
  }
  return 1.0;
}
template<> double coefGemmSOL<double>(int mathMode, int major, int minor, int algorithm) {
  if (major >= 8) {
    return 1.0; // SM80 - using dmma
  } else if(major == 7) {
    return 0.5; // SM70+
  } else if(major == 6) {
    if(minor == 0)
      return 0.5; // SM60
    else
      return 1.0/32.0; // SM61+
  } else if (major == 5) {
    return 1.0/32.0;
  } else {
    return 1.0/3.0;
  }
}
template<> double coefGemmSOL<__half>(int mathMode, int major, int minor, int algorithm) {
  if(major >= 7) {
    //get mathMode or algorithm value
    if(mathMode == 1 || algorithm >= 99) {
      return 8.0;//1024.0 / 128.0;
    } else {
      return 2.0; // SM70 - using hfma2
    }
  }else if(major == 6) {
    if(minor == 0)
      return 2.0; // SM60 - using hfma2
    else
      return 1.0/64.0; // SM61+
  } else {
    return 0.0;
  }
}
/***
template<> double coefGemmSOL<__bfloat16>(int mathMode, int major, int minor, int algorithm) {
  if(major >= 8) {
    return coefGemmSOL<__half>(mathMode, major, minor, algorithm); // assuming same as for half
  } else {
    return 1.0; // using ffma
  }
}
****/

template<> double coefGemmSOL<int>(int mathMode, int major, int minor, int algorithm) {
  if(major >= 7) {
    return 4.0;
  } else if(major == 6) {
    if(minor == 1)
      return 4.0; // SM61 - using dp4a
    else
      return 1.0;
  } else {
    return 1.0;
  }
}
template<> double coefGemmSOL<int8_t>(int mathMode, int major, int minor, int algorithm) {
  return 1.0;
}
/*
template<> double coefGemmSOL<cuInt8Complex>(int mathMode, int major, int minor, int algorithm) {
    return coefGemmSOL<int8_t>(mathMode, major, minor, algorithm);
}*/
template<> double coefGemmSOL<cuComplex>(int mathMode, int major, int minor, int algorithm) {
  return coefGemmSOL<float>(mathMode, major, minor, algorithm);
}
template<> double coefGemmSOL<cuDoubleComplex>(int mathMode, int major, int minor, int algorithm) {
  return coefGemmSOL<double>(mathMode, major, minor, algorithm);
}

template<typename T_MATH> char gemmType();
template<> char gemmType<float>() { return 's'; }
template<> char gemmType<double>() { return 'd'; }
template<> char gemmType<__half>() { return 'h'; }
/***
template<> char gemmType<__bfloat16>() { return 't'; }
***/
template<> char gemmType<int>() { return 'i'; }
template<> char gemmType<int8_t>() { return 'b'; }
template<> char gemmType<cuComplex>() { return 'c'; }
template<> char gemmType<cuDoubleComplex>() { return 'z'; }

template<typename T_MATH> double flopsCoef() { return 2.0; }
template<> double flopsCoef<cuComplex>() { return 8.0; }
template<> double flopsCoef<cuDoubleComplex>() { return 8.0; }

// printSOL prints various Speed of Light information (maximum Gflops, achieved Gflops)
//  - computeSeconds: Number of seconds consumed during compute time
//  - iterations    : Number of iterations (times the gemm was repeated)
//  - m, n, k       : Dimensions of the gemm
template<typename T_MATH>
void printGemmSOL(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm)
{
  // Header for all SOL information
  printf("^^^^ %cgemm SOL  : ", gemmType<T_MATH>());

  // Get # of flops
  double theoryFlops = (double)iterations * flopsCoef<T_MATH>() * (double)m * (double)n * (double)k;

  // Calculate throughput obtained (FLOPs / seconds)
  double realThroughput =  (double)theoryFlops / (double)computeSeconds;

  // Get currently selected device
  int device_id;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    printf("Calculation Failure: Get Device Failure");
    return;
  }

  // Query device properties
  struct cudaDeviceProp prop;
  if (cudaGetDeviceProperties( &prop, device_id ) != cudaSuccess) {
    printf("Calculation Failure: Device Query Failure");
    return;
  }

  // Set theoretical throughput to 0 at first (will be set later based on architecture)
  double theoryThroughput = 0;

  assert((prop.major == 3) || (prop.major == 5) || (prop.major == 6) || (prop.major == 7) || (prop.major == 8));
  if(prop.major == 8) {
    theoryThroughput = 2 * 64  * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
  } else if(prop.major == 7) {
    theoryThroughput = 2 * 64  * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
  } else if(prop.major == 6) { // On Pascal, we have 64 or 128 FMAs per SM per clock
    if(prop.minor == 0) { // SM60 GP100
      theoryThroughput = 2 * 64  * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
    } else { // SM61+ GP102+
      theoryThroughput = 2 * 128 * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
    }
  }
  // If Maxwell, we can compute 128FMAs per SM per clock
  else if(prop.major > 3){
    theoryThroughput = 2 * 128 * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
  }
  // If Kepler, we can compute 192FMAs per SM per clock
  else{
    theoryThroughput = 2 * 192 * (double)prop.multiProcessorCount * (double)prop.clockRate*1e3;
  }
  // Correct for non-sgemm flops count, depending om the architecture
  theoryThroughput *= coefGemmSOL<T_MATH>(mathMode, prop.major, prop.minor, algorithm);

  // Output final SOL info
  printf("theoretical Gflops = %f, measured Gflops = %f (%f%%)\n", theoryThroughput*1e-9, realThroughput*1e-9, 100*(realThroughput / theoryThroughput));
}

template void printGemmSOL<float>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
template void printGemmSOL<double>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
template void printGemmSOL<__half>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
/**
template void printGemmSOL<__bfloat16>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
***/
template void printGemmSOL<int>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
template void printGemmSOL<int8_t>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
template void printGemmSOL<cuComplex>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);
template void printGemmSOL<cuDoubleComplex>(int mathMode, double computeSeconds, int iterations, int m, int n, int k, int algorithm);



