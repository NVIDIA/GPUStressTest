#include "common_header.h"

cudaError_t
get_device_version(int &device_version)
{
  int device;
  struct cudaDeviceProp properties;
  cudaError_t error;

  error = cudaGetDevice (&device);
  if (error != cudaSuccess) {
    fprintf (stdout,"failed to get device cudaError=%d\n", error);
    return error;
  }
    
  error = cudaGetDeviceProperties (&properties, device);
  if (error != cudaSuccess) {
    fprintf (stdout,"failed to get properties cudaError=%d\n", error);
    return error;
  } else {        
    device_version =  properties.major * 100 + properties.minor * 10;
  }
  return error;
}


