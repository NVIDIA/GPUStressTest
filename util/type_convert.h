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


#pragma once
static int roundoff(int v, int d) { return (v + d - 1) / d * d; }

template <typename T_ELEM>
__inline__ __device__ __host__ T_ELEM cuGet(float);
template <>
__inline__ __device__ __host__ float cuGet<float>(float x) {
  return float(x);
}

template <>
__inline__ __device__ __host__ int cuGet<int>(float x) {
  return (int)(x);
}
template <>
__inline__ __device__ __host__ __half cuGet<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__inline__ __device__ __host__ __nv_bfloat16 cuGet<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <>
__inline__ __device__ __host__ cuComplex cuGet<cuComplex>(float x) {
  return (make_cuComplex(float(x), 0.0f));
}

template <>
__inline__ __device__ __host__ cuDoubleComplex cuGet<cuDoubleComplex>(float x) {
  return (make_cuDoubleComplex(double(x), 0.0));
}

template <>
__inline__ __device__ __host__ double cuGet<double>(float x) {
  return double(x);
}

template <>
__inline__ __device__ __host__ int8_t cuGet<int8_t>(float x) {
  return (int8_t)x;
}
template <>
__inline__ __device__ __host__ unsigned char cuGet<unsigned char>(float x) {
  return (unsigned char)x;
}
template <>
__inline__ __device__ __host__ __nv_fp8_e4m3 cuGet<__nv_fp8_e4m3>(float x) {
  return (__nv_fp8_e4m3)x;
}

static __inline__ unsigned floatAsUInt(float x) {
  volatile union {
    float f;
    unsigned i;
  } xx;
  xx.f = x;
  return xx.i;
}

static __inline__ unsigned long long doubleAsULL(double x) {
  volatile union {
    double f;
    unsigned long long i;
  } xx;
  xx.f = x;
  return xx.i;
}

/* Fma */
static __inline__ __device__ __host__ int cuFma(int x, int y, int d) {
  return ((x * y) + d);
}

static __inline__ __device__ __host__ float cuFma(float x, float y, float d) {
  return ((x * y) + d);
}

static __inline__ __device__ __host__ double cuFma(double x, double y,
                                                   double d) {
  return ((x * y) + d);
}

static __inline__ __device__ __host__ cuComplex cuFma(cuComplex x, cuComplex y,
                                                      cuComplex d) {
  return (cuCfmaf(x, y, d));
}

static __inline__ __device__ __host__ cuDoubleComplex cuFma(cuDoubleComplex x,
                                                            cuDoubleComplex y,
                                                            cuDoubleComplex d) {
  return (cuCfma(x, y, d));
}

static __inline__ __device__ __host__ half cuFma(half a, half b, half c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return (__hfma(a, b, c));
#else
  return cuGet<half>(cuGet<float>(a) * cuGet<float>(b) + cuGet<float>(c));
#endif
}

template <typename T>
__inline__ __device__ __host__ T cuMakeNaN() {
  T val;
  memset(&val, 0xff, sizeof(val));
  return val;
}

/*--------------------------------------------------------------------------------------------*/
template <typename T_ELEM>
__inline__ __device__ __host__ T_ELEM cuGet(double, double);
template <>
__inline__ __device__ __host__ float cuGet<float>(double x, double y) {
  return float(x);
}

template <>
__inline__ __device__ __host__ double cuGet<double>(double x, double y) {
  return double(x);
}

template <>
__inline__ __device__ __host__ cuComplex cuGet<cuComplex>(double x, double y) {
  return (make_cuComplex(float(x), float(y)));
}

template <>
__inline__ __device__ __host__ cuDoubleComplex
cuGet<cuDoubleComplex>(double x, double y) {
  return (make_cuDoubleComplex(double(x), double(y)));
}
template <>
__inline__ __device__ __host__ __half cuGet<__half>(double x, double y) {
  return cuGet<__half>(x);
}

template <>
__inline__ __device__ __host__ int8_t cuGet<int8_t>(double x, double y) {
  return ((int8_t)((int)x));
}

template <>
__inline__ __device__ __host__ unsigned char cuGet<unsigned char>(double x,
                                                                  double y) {
  return ((unsigned char)((int)x));
}

template <>
__inline__ __device__ __host__ int cuGet<int>(double x, double y) {
  return ((int)x);
}

template <>
__inline__ __device__ __host__ __nv_bfloat16 cuGet<__nv_bfloat16>(double x,
                                                                  double y) {
  return cuGet<__nv_bfloat16>(x);
}

template <>
__inline__ __device__ __host__ __nv_fp8_e4m3 cuGet<__nv_fp8_e4m3>(double x,
                                                                  double y) {
  return cuGet<__nv_fp8_e4m3>(x);
}

template <typename T, typename... Candidates>
struct is_same_type;
template <typename T, typename U>
struct is_same_type<T, U> {
  static const bool value = false;
};
template <typename T>
struct is_same_type<T, T> {
  static const bool value = true;
};
template <typename T, typename C, typename... Candidates>
struct is_same_type<T, C, Candidates...> {
  static const bool value =
      is_same_type<T, C>::value || is_same_type<T, Candidates...>::value;
};

template <typename T_ELEM_OUT, typename T_SCALE>
struct biasType {
  typedef T_ELEM_OUT type;
};

template <>
struct biasType<int, float> {
  typedef float type;
};
template <>
struct biasType<int8_t, float> {
  typedef float type;
};

template <typename T_ELEM_IN_A, typename T_ELEM_IN_B, typename T_ELEM_IN_C,
          typename T_ELEM_OUT, typename T_SCALE, typename = void>
struct biasTypeExtended {
  using type = typename biasType<T_ELEM_OUT, T_SCALE>::type;
};

template <typename T_ELEM_IN_A, typename T_ELEM_IN_B, typename T_ELEM_IN_C,
          typename T_ELEM_OUT, typename T_SCALE>
struct biasTypeExtended<
    T_ELEM_IN_A, T_ELEM_IN_B, T_ELEM_IN_C, T_ELEM_OUT, T_SCALE,
    typename std::enable_if<
        is_same_type<T_ELEM_IN_A, __nv_fp8_e4m3>::value ||
        is_same_type<T_ELEM_IN_B, __nv_fp8_e4m3>::value>::type> {
  using type = __nv_bfloat16;
};

template <cudaDataType_t t>
struct CudaTypeEnumTraits;

#define MAKE_TYPE_TRAITS(type_, type_enum)   template <>                                struct CudaTypeEnumTraits<type_enum> {       typedef type_ type;                      }

MAKE_TYPE_TRAITS(float, CUDA_R_32F);
MAKE_TYPE_TRAITS(cuComplex, CUDA_C_32F);
MAKE_TYPE_TRAITS(double, CUDA_R_64F);
MAKE_TYPE_TRAITS(cuDoubleComplex, CUDA_C_64F);
MAKE_TYPE_TRAITS(int8_t, CUDA_R_8I);
MAKE_TYPE_TRAITS(__nv_bfloat16, CUDA_R_16BF);
MAKE_TYPE_TRAITS(int32_t, CUDA_R_32I);
MAKE_TYPE_TRAITS(__nv_fp8_e4m3, CUDA_R_8F_E4M3);
MAKE_TYPE_TRAITS(__half, CUDA_R_16F);

