#pragma once

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
  *
  */

#include <stdio.h>
#include <string>

/* WGST specific defines */
#define MAX_NUM_GPUS 32
#define NUM_TESTS 5
#define TEST_WAIT_TIME 600

#ifdef __linux__ 
#else
#include <Windows.h>
#endif

class WGST {

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

    enum test_suite {T4, A100};

    WGST(const test_suite gpu) {
        switch (gpu) {
        case T4:
            init_t4();
            break;
        case A100:
            init_a100C();
            break;
        }
    }

    ~WGST() {}

    /*
     * Values set as multiples of A100 Benchmark Guide values
     * for broader GPU memory footprint coverage
     */
    WGST()
    {
        init_t4();
    }
    
    void dump_test_args(const int t_num) {
        printf("stress_tests[%d].test_name %s P_arg %s\n", t_num, stress_tests[t_num].test_name, stress_tests[t_num].P_arg.c_str());
        return;
    }

private: 
    void init_a100() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 32768;
        stress_tests[0].n_arg = 13824;
        stress_tests[0].k_arg = 65536;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 36864;
        stress_tests[1].n_arg = 27648;
        stress_tests[1].k_arg = 88304;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 32768;
        stress_tests[2].n_arg = 13824;
        stress_tests[2].k_arg = 65536;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 13824;
        stress_tests[3].n_arg = 8192;
        stress_tests[3].k_arg = 65536;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 20736;
        stress_tests[4].n_arg = 12288;
        stress_tests[4].k_arg = 98304;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

    void init_a100A() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 81218;
        stress_tests[0].n_arg = 34263;
        stress_tests[0].k_arg = 162437;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 58982;
        stress_tests[1].n_arg = 44236;
        stress_tests[1].k_arg = 157286;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 52428;
        stress_tests[2].n_arg = 22118;
        stress_tests[2].k_arg = 104858;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 23506;
        stress_tests[3].n_arg = 13730;
        stress_tests[3].k_arg = 123830;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 33178;
        stress_tests[4].n_arg = 19660;
        stress_tests[4].k_arg = 157286;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

    void init_a100B() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 32768;
        stress_tests[0].n_arg = 13824;
        stress_tests[0].k_arg = 65536;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 16384;
        stress_tests[1].n_arg = 13824;
        stress_tests[1].k_arg = 49152;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 8192;
        stress_tests[2].n_arg = 6912;
        stress_tests[2].k_arg = 32768;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 8192;
        stress_tests[3].n_arg = 6912;
        stress_tests[3].k_arg = 32768;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 8192;
        stress_tests[4].n_arg = 6912;
        stress_tests[4].k_arg = 32768;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }


    void init_a100C() {
        stress_tests[0].test_name = "INT8";
        stress_tests[0].test_state = 0;
        stress_tests[0].P_arg = "bisb_imma";
        stress_tests[0].m_arg = 81218;
        stress_tests[0].n_arg = 34263;
        stress_tests[0].k_arg = 162437;
        stress_tests[0].ta_arg = 1;
        stress_tests[0].tb_arg = 0;
        stress_tests[0].B_arg = 0;

        stress_tests[1].test_name = "FP16";
        stress_tests[1].test_state = 0;
        stress_tests[1].P_arg = "hsh";
        stress_tests[1].m_arg = 58982;
        stress_tests[1].n_arg = 44236;
        stress_tests[1].k_arg = 157286;
        stress_tests[1].ta_arg = 0;
        stress_tests[1].tb_arg = 1;
        stress_tests[1].B_arg = 0;

        stress_tests[2].test_name = "TF32";
        stress_tests[2].test_state = 0;
        stress_tests[2].P_arg = "sss_fast_tf32";
        stress_tests[2].m_arg = 52428;
        stress_tests[2].n_arg = 22118;
        stress_tests[2].k_arg = 104858;
        stress_tests[2].ta_arg = 1;
        stress_tests[2].tb_arg = 0;
        stress_tests[2].B_arg = 0;

        stress_tests[3].test_name = "FP64";
        stress_tests[3].test_state = 0;
        stress_tests[3].P_arg = "ddd";
        stress_tests[3].m_arg = 23506;
        stress_tests[3].n_arg = 13730;
        stress_tests[3].k_arg = 123830;
        stress_tests[3].ta_arg = 0;
        stress_tests[3].tb_arg = 1;
        stress_tests[3].B_arg = 0;

        stress_tests[4].test_name = "FP32";
        stress_tests[4].test_state = 0;
        stress_tests[4].P_arg = "sss";
        stress_tests[4].m_arg = 33178;
        stress_tests[4].n_arg = 19660;
        stress_tests[4].k_arg = 125286;
        stress_tests[4].ta_arg = 0;
        stress_tests[4].tb_arg = 1;
        stress_tests[4].B_arg = 0;
    }

   

    void init_t4() {
        stress_tests[0].test_name = "FP16"; 
        stress_tests[0].test_state = 0; 
        stress_tests[0].P_arg = "hsh"; 
        stress_tests[0].m_arg = 36864; 
        stress_tests[0].n_arg = 27648; 
        stress_tests[0].k_arg = 98304; 
        stress_tests[0].ta_arg = 0; 
        stress_tests[0].tb_arg = 1; 
        stress_tests[0].B_arg = 0; 
 
        stress_tests[1].test_name = "C32"; 
        stress_tests[1].test_state = 0; 
        stress_tests[1].P_arg = "ccc"; 
        stress_tests[1].m_arg = 18432; 
        stress_tests[1].n_arg = 13824; 
        stress_tests[1].k_arg = 49170; 
        stress_tests[1].ta_arg = 0; 
        stress_tests[1].tb_arg = 1; 
        stress_tests[1].B_arg = 0; 
 
        stress_tests[2].test_name = "FP32"; 
        stress_tests[2].test_state = 0; 
        stress_tests[2].P_arg = "hhh"; 
        stress_tests[2].m_arg = 36864; 
        stress_tests[2].n_arg = 27648; 
        stress_tests[2].k_arg = 98304; 
        stress_tests[2].ta_arg = 0; 
        stress_tests[2].tb_arg = 1; 
        stress_tests[2].B_arg = 0; 
 
        stress_tests[3].test_name = "FP64"; 
        stress_tests[3].test_state = 0; 
        stress_tests[3].P_arg = "zzz"; 
        stress_tests[3].m_arg = 9216; 
        stress_tests[3].n_arg = 6912; 
        stress_tests[3].k_arg = 24585; 
        stress_tests[3].ta_arg = 0; 
        stress_tests[3].tb_arg = 1; 
        stress_tests[3].B_arg = 0; 
     
        stress_tests[4].test_name = "FP32"; 
        stress_tests[4].test_state = 0; 
        stress_tests[4].P_arg = "hss"; 
        stress_tests[4].m_arg = 18432; 
        stress_tests[4].n_arg = 13824; 
        stress_tests[4].k_arg = 49170; 
        stress_tests[4].ta_arg = 0; 
        stress_tests[4].tb_arg = 1; 
        stress_tests[4].B_arg = 0;
    }

};




