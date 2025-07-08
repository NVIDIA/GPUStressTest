#! /bin/bash

input=${1:-gst.out}

cat $input | awk ' 
BEGIN {
  M = 0 + 0
  N = 0 + 0 
  K = 0 + 0
  A = 0 + 0
  B = 0 + 0
  C = 0 + 0
  GB=1073741824
}
/Initilizing/ {gpu=$2} 
/memgb:/ {targetgb=$5}
/STARTING TEST/ {test=$5}
/MATRIX SIZES/ {A=$3; B=$4; C=$5}
/matrixSizeA/ {M=$2; N=$4; K=$6; OP=$8}
 
/PASSED/ {
  size=A * M + B * N + C * K + (OP) ? C * K: 0;
  ratio=(size / (targetgb * GB)) + 0.0
  if (ratio > 0.80 && ratio  < 0.95) good=1; else good=0
  printf("gpu %s test %s m %d n %d k %d A %d B %d C %d OP %d targetgb %d GB %d size %d ratio %2.3f good %d\n", gpu, test, M, N, K, A, B, C, OP, targetgb, (targetgb * GB), size, ratio, good)
} '







