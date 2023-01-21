#pragma once
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


 /*
  * Define the BLAS tests and assocaited args as per the A100 Benchmark Guide.
  * We'll reference this data out of the for loop in main.cu to run each of
  * 5 tests on each GPU.
  *
  * For reference, the options from -h are listed here.
  *
  *    Test-specific options :
  *     -P={sss,sss_fast_tf32,ccc,hhh,ddd,zzz,hsh,hss,bisb_imma,bii_imma}
  *     sss: input type: float, math type: float, scale type: float, output type: float
  *     sss_fast_tf32: input type: float, math type: float, scale type: float, output type: float
  *     ccc: input type: complex, math type: complex, scale type: complex, output type: complex
  *     hhh: input type: half, math type: half, scale type: half, output type: half
  *     ddd: input type: double, math type: double, scale type: double, output type: double
  *     zzz: input type: double complex, math type: double complex, scale type: double complex, output type: double complex
  *     hsh: input type: half, math type: float, output type: half , scale type: float
  *     hss: input type: half, math type: float, output type: float, scale type: float
  *     bisb_imma: input type: int8, math type: int32, scale type: float, output type: int8
  *     bii_imma: input type: int8, math type: int32, scale type: int32, output type: int32
  *     -m=<int>  : number of rows of A and C
  *     -n=<int>  : number of columns of B and C
  *     -k=<int>  : number of columns of A and rows of B
  *     -A=<float>  : value of alpha
  *     -B=<float>  : value of beta
  *     -T=<int> : run N times back to back  , good for power consumption, no results checking
  *     -lda=<int> : leading dimension of A , m by default
  *     -ldb<number> : leading dimension of B , k by default
  *     -ldc<number> : leading dimension of C , m by default
  *     -ta= op(A) {0=no transpose, 1=transpose, 2=hermitian}
  *     -tb= op(B) {0=no transpose, 1=transpose, 2=hermitian}
  *     -dv=<int> select a GPU number to test
  *
  */

#include <stdio.h>
#include <string>

/* GST specific defines */
#define MAX_NUM_GPUS 32
#define NUM_TESTS 5
#define TEST_WAIT_TIME 600

#ifdef __linux__ 
#else
#include <Windows.h>
#endif

class GST {

public: 

    struct stress_test_args {
        const char* test_name = "undefined";
        int test_state = 0;
        std::string P_arg;
        int m_arg = 0;
        int n_arg = 0;
        int k_arg = 0;
        int ta_arg = 0;
        int tb_arg = 0;
        int B_arg = 0;
    };

    struct stress_test_args stress_tests[NUM_TESTS];

    enum test_suite {T4, A100_40, A100_80, K80, M60, P40, P100, H100, V100_16, V100_32, Generic};

    GST(const test_suite gpu) {
        switch (gpu) {
        case T4:
        case P100: // same as T4
        case V100_16: // same as T4
            init_t4();
            break;
        case A100_40:
            init_a100_40();
            break;
        case H100: // same as a100_80
        case A100_80:
            init_a100_80();
            break;
        case K80:
            init_k80();
            break;
        case P40:
            init_p40();
            break;
        case V100_32:
            init_v100_32();
            break;
        case Generic:
        case M60: //same as generic
            init_generic();
            break;
        }
    }

    ~GST() {}

    /*
     * Values set as multiples of A100 Benchmark Guide values
     * for broader GPU memory footprint coverage
     */
    GST()
    {
        init_generic();
    }
    
    void dump_test_args(const int t_num) {
        printf("stress_tests[%d].test_name %s P_arg %s\n", t_num, stress_tests[t_num].test_name, stress_tests[t_num].P_arg.c_str());
        return;
    }

private: 

    void init_a100_40() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 81218;
        stress_tests[0].n_arg = 34263;
        stress_tests[0].k_arg = 322437;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 59982;
        stress_tests[1].n_arg = 49336;
        stress_tests[1].k_arg = 157286;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 210712;
        stress_tests[2].n_arg = 68472;
        stress_tests[2].k_arg = 17216;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 23506;
        stress_tests[3].n_arg = 10730;
        stress_tests[3].k_arg = 65830;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 225712;
        stress_tests[4].n_arg = 58640;
        stress_tests[4].k_arg = 15072;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

    void init_v100_16() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 39864;
        stress_tests[0].n_arg = 28648;
        stress_tests[0].k_arg = 98304;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 14432;
        stress_tests[1].n_arg = 19424;
        stress_tests[1].k_arg = 49170;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 16064;
        stress_tests[2].n_arg = 20608;
        stress_tests[2].k_arg = 202750;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 12216;
        stress_tests[3].n_arg = 9912;
        stress_tests[3].k_arg = 40085;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 18402;
        stress_tests[4].n_arg = 13704;
        stress_tests[4].k_arg = 240070;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }


    void init_v100_32() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 20728;
        stress_tests[0].n_arg = 55296;
        stress_tests[0].k_arg = 196608;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 12264;
        stress_tests[1].n_arg = 28250;
        stress_tests[1].k_arg = 91340;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 40864;
        stress_tests[2].n_arg = 25048;
        stress_tests[2].k_arg = 230500;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 30432;
        stress_tests[3].n_arg = 30000;
        stress_tests[3].k_arg = 17170;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 31564;
        stress_tests[4].n_arg = 27048;
        stress_tests[4].k_arg = 225022;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }


    void init_a100_80() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 5218;
        stress_tests[0].n_arg = 263;
        stress_tests[0].k_arg = 2437;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 5928;
        stress_tests[1].n_arg = 6944;
        stress_tests[1].k_arg = 2144;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

/*
        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 209712;
        stress_tests[2].n_arg = 68472;
        stress_tests[2].k_arg = 90432;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;
*/


        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 105712;
        stress_tests[2].n_arg = 34472;
        stress_tests[2].k_arg = 45432;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 44024;
        stress_tests[3].n_arg = 30920;
        stress_tests[3].k_arg = 48320;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

/*
        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 132712;
        stress_tests[4].n_arg = 58640;
        stress_tests[4].k_arg = 170144;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
*/

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 105712;
        stress_tests[4].n_arg = 34472;
        stress_tests[4].k_arg = 45432;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;

    }


    void init_t4() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 31864;
        stress_tests[0].n_arg = 38648;
        stress_tests[0].k_arg = 88304;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 11432;
        stress_tests[1].n_arg = 16424;
        stress_tests[1].k_arg = 61000;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 16064;
        stress_tests[2].n_arg = 20608;
        stress_tests[2].k_arg = 192750;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 12016;
        stress_tests[3].n_arg = 9012;
        stress_tests[3].k_arg = 40085;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 18902;
        stress_tests[4].n_arg = 15904;
        stress_tests[4].k_arg = 200070;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

    void init_k80() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 22960;
        stress_tests[0].n_arg = 11170;
        stress_tests[0].k_arg = 150890;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 11980;
        stress_tests[1].n_arg = 8985;
        stress_tests[1].k_arg = 60060;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 20060;
        stress_tests[2].n_arg = 16970;
        stress_tests[2].k_arg = 135490;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 5990;
        stress_tests[3].n_arg = 4092;
        stress_tests[3].k_arg = 64080;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 12980;
        stress_tests[4].n_arg = 8985;
        stress_tests[4].k_arg = 240960;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }


    void init_p40() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 49766;
        stress_tests[0].n_arg = 37260;
        stress_tests[0].k_arg = 102710;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 24883;
        stress_tests[1].n_arg = 18525;
        stress_tests[1].k_arg = 51979;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 49766;
        stress_tests[2].n_arg = 37260;
        stress_tests[2].k_arg = 100958;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 12441;
        stress_tests[3].n_arg = 9331;
        stress_tests[3].k_arg = 58189;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 20883;
        stress_tests[4].n_arg = 14525;
        stress_tests[4].k_arg = 299379;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

    void init_generic() {
        stress_tests[0].test_name = "FP16";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "hsh";
        stress_tests[0].m_arg = 18432;
        stress_tests[0].n_arg = 18824;
        stress_tests[0].k_arg = 100152;
        stress_tests[0].ta_arg = 0;
        stress_tests[0].tb_arg = 1;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "C32";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "ccc";
        stress_tests[1].m_arg = 9816;
        stress_tests[1].n_arg = 8012;
        stress_tests[1].k_arg = 50885;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "FP32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "hhh";
        stress_tests[2].m_arg = 15832;
        stress_tests[2].n_arg = 15824;
        stress_tests[2].k_arg = 120052;
        stress_tests[2].ta_arg = 0;
        stress_tests[2].tb_arg = 1;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "zzz";
        stress_tests[3].m_arg = 4608;
        stress_tests[3].n_arg = 3456;
        stress_tests[3].k_arg = 60291;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "hss";
        stress_tests[4].m_arg = 18832;
        stress_tests[4].n_arg = 13824;
        stress_tests[4].k_arg = 115070;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }


};




