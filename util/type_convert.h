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
static int
roundoff(int v, int d) {
    return (v + d - 1) / d * d;
}

template <typename T_ELEM> __inline__ __device__ __host__  T_ELEM cuGet (float );
template <> __inline__ __device__ __host__  float cuGet<float >(float x)
{
  return float(x);
}

template <> __inline__ __device__ __host__  int cuGet<int >(float x)
{
  return (int)(x);
}
template <> __inline__ __device__ __host__  __half cuGet<__half >(float x)
{
  return __float2half_rn( x );
}
/****
template <> __inline__ __device__ __host__  __bfloat16 cuGet<__bfloat16 >(float x)
{
  return __float2bfloat16_rn( x );
}
****/

template <> __inline__ __device__ __host__   cuComplex cuGet<cuComplex>(float x)
{
  return (make_cuComplex( float(x), 0.0f ));
}

template <> __inline__ __device__ __host__   cuDoubleComplex  cuGet<cuDoubleComplex>(float x)
{
  return (make_cuDoubleComplex( double(x), 0.0 ));
}

template <> __inline__ __device__ __host__  double cuGet<double>(float x)
{
  return double(x);
}

template <> __inline__ __device__ __host__  int8_t cuGet<int8_t >(float x)
{
  return (int8_t)x;
}
template <> __inline__ __device__ __host__  unsigned char cuGet<unsigned char >(float x)
{
  return (unsigned char)x;
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



