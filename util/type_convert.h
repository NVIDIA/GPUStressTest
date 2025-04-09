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
#pragma once

/* definition of int8 complex */
typedef struct __align__(2) {
  int8_t x;
  int8_t y;
}
cuInt8Complex;

typedef __half2 cublasHalfComplex;
typedef __nv_bfloat162 cublasBfloat16Complex;

namespace custd {

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

template <class T, T v>
struct integral_constant {
  static const T value = v;
};
template <bool B>
struct bool_constant : integral_constant<bool, B> {};
struct true_type : bool_constant<true> {};
struct false_type : bool_constant<false> {};


template <typename T>
struct is_complex : bool_constant<is_same_type<T, cuComplex>::value ||              //
                                  is_same_type<T, cuDoubleComplex>::value ||        //
                                  is_same_type<T, cublasHalfComplex>::value ||      //
                                  is_same_type<T, cublasBfloat16Complex>::value ||  //
                                  is_same_type<T, cuInt8Complex>::value> {};

template <typename T>
struct is_integer : bool_constant<is_same_type<T, int>::value ||       //
                                  is_same_type<T, unsigned>::value ||  //
                                  is_same_type<T, uint64_t>::value ||  //
                                  is_same_type<T, int64_t>::value ||   //
                                  is_same_type<T, uint16_t>::value ||  //
                                  is_same_type<T, int16_t>::value ||   //
                                  is_same_type<T, uint8_t>::value ||   //
                                  is_same_type<T, int8_t>::value ||    //
                                  is_same_type<T, cuInt8Complex>::value> {};

template <typename T>
struct math_traits {
  using MathType = T;
};

template <>
struct math_traits<__half> {
  using MathType = float;
};

template <>
struct math_traits<__nv_bfloat16> {
  using MathType = float;
};

template <>
struct math_traits<__nv_fp8_e4m3> {
  using MathType = float;
};

template <>
struct math_traits<__nv_fp8_e5m2> {
  using MathType = float;
};

template <>
struct math_traits<__nv_fp8_e8m0> {
  using MathType = float;
};

template <>
struct math_traits<__nv_fp4_e2m1> {
  using MathType = float;
};


} // namespace custd

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
template <>
__inline__ __device__ __host__ __nv_fp4_e2m1 cuGet<__nv_fp4_e2m1>(float x) {
  return (__nv_fp4_e2m1)x;
}
template <>
__inline__ __device__ __host__ __nv_fp8_e8m0 cuGet<__nv_fp8_e8m0>(float x) {
  return (__nv_fp8_e8m0)x;
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

template <>
__inline__ __device__ __host__ __nv_fp4_e2m1 cuGet<__nv_fp4_e2m1>(double x,
                                                                  double y) {
  return cuGet<__nv_fp4_e2m1>(x);
}

template <>
__inline__ __device__ __host__ __nv_fp8_e8m0 cuGet<__nv_fp8_e8m0>(double x,
                                                                  double y) {
  return cuGet<__nv_fp8_e8m0>(x);
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

#define MAKE_TYPE_TRAITS(type_, type_enum) \
  template <>                              \
  struct CudaTypeEnumTraits<type_enum> {   \
    typedef type_ type;                    \
  }

MAKE_TYPE_TRAITS(float, CUDA_R_32F);
MAKE_TYPE_TRAITS(cuComplex, CUDA_C_32F);
MAKE_TYPE_TRAITS(double, CUDA_R_64F);
MAKE_TYPE_TRAITS(cuDoubleComplex, CUDA_C_64F);
MAKE_TYPE_TRAITS(int8_t, CUDA_R_8I);
MAKE_TYPE_TRAITS(__nv_bfloat16, CUDA_R_16BF);
MAKE_TYPE_TRAITS(int32_t, CUDA_R_32I);
MAKE_TYPE_TRAITS(__nv_fp8_e4m3, CUDA_R_8F_E4M3);
MAKE_TYPE_TRAITS(__half, CUDA_R_16F);
MAKE_TYPE_TRAITS(__nv_fp4_e2m1, CUDA_R_4F_E2M1);


/*------------------------------------------------------------------------------------------------*/
/* Multiplication */
template <typename T>
inline __device__ __host__ T cuMul(T x, T y) {
  using M = typename custd::math_traits<T>::MathType;
  return T(M(x) * M(y));
}
template <>
inline __device__ __host__ cuComplex cuMul(cuComplex x, cuComplex y) {
  return (cuCmulf(x, y));
}
template <>
inline __device__ __host__ cuDoubleComplex cuMul(cuDoubleComplex x, cuDoubleComplex y) {
  return (cuCmul(x, y));
}
template <>
inline __device__ __host__ __half cuMul(__half x, __half y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul(x, y);
#else
  return cuGet<__half>(cuGet<float>(x) * cuGet<float>(y));
#endif
}
template <>
inline __device__ __host__ __nv_bfloat16 cuMul(__nv_bfloat16 x, __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hmul(x, y);
#else
  return cuGet<__nv_bfloat16>(cuGet<float>(x) * cuGet<float>(y));
#endif
}

/*------------------------------------------------------------------------------------------------*/
/* Addition */
template <typename T>
inline __device__ __host__ typename std::enable_if<!custd::is_complex<T>::value, T>::type cuAdd(T x, T y) {
  using M = typename custd::math_traits<T>::MathType;
  return M(x) + M(y);
}
template <typename T>
inline __device__ __host__ typename std::enable_if<custd::is_complex<T>::value, T>::type cuAdd(T x, T y) {
  return {cuAdd(x.x, y.x), cuAdd(x.y, y.y)};
}
// Keeping specialization in case future architectures have something special for regular complex data types
template <>
inline __device__ __host__ cuComplex cuAdd(cuComplex x, cuComplex y) {
  return cuCaddf(x, y);
}
template <>
inline __device__ __host__ cuDoubleComplex cuAdd(cuDoubleComplex x, cuDoubleComplex y) {
  return cuCadd(x, y);
}
template <>
inline __device__ __host__ __half cuAdd(__half a, __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return (__hadd(a, b));
#else
  return cuGet<__half>(cuGet<float>(a) + cuGet<float>(b));
#endif
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
inline __device__ __host__ __nv_bfloat16 cuAdd(__nv_bfloat16 a, __nv_bfloat16 b) {
  return (__hadd(a, b));
}
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template <>
inline __device__ __host__ __half2 cuAdd(__half2 a, __half2 b) {
  return (__hadd2(a, b));
}
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
inline __device__ __host__ __nv_bfloat162 cuAdd(__nv_bfloat162 a, __nv_bfloat162 b) {
  return (__hadd2(a, b));
}
#endif
