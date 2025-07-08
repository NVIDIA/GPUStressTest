#! /bin/bash

input=${1:-gst.out}

cat $input | awk ' 
BEGIN {
  Sa = 0 + 0
  Sb = 0 + 0 
  Sc = 0 + 0
  A = 0 + 0
  B = 0 + 0
  C = 0 + 0
  GB=1073741824
}
/Initilizing/ {gpu=$2} 
/memgb:/ {targetgb=$5}
/STARTING TEST/ {test=$5}
/MATRIX SIZES/ {A=$3; B=$4; C=$5}
/matrixSizeA/ {Sa=$2; Sb=$4; Sc=$6; OP=$8}
 
/PASSED/ {
  size=A * Sa + B * Sb + C * Sc + (OP) ? C * Sc: 0;
  ratio=(size / (targetgb * GB)) + 0.0
  if (ratio > 0.80 && ratio  < 0.95) good=1; else good=0
  printf("gpu %s test %s Sa %d Sb %d Sc %d A %d B %d C %d OP %d targetgb %d GB %d size %d ratio %2.3f good %d\n", gpu, test, Sa, Sb, Sc, A, B, C, OP, targetgb, (targetgb * GB), size, ratio, good)
} '







