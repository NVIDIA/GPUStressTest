#! /bin/bash

input=${1:-gst.out}

# Empiriacly tge FP64 test appares to allocate 16 bytes per matrix element
# is this actually C64 ?????  
# ALso the XX32 are showing up as 2 byes 
cat $input | awk ' 
BEGIN {
  GB=1073741824
  size["INT8"]=1
  size["FP16"]=2
  size["TF32"]=2
  size["FP64"]=16
  size["FP32"]=2
  size["C32"]=8
}
/Initilizing/ {gpu=$2} 
/memgb:/ {targetgb=$5}
/STARTING TEST/ {test=$5}
/matrixSizeA/ {A=$3; B=$5; C=$7}
/PASSED/ {
  ratio=((A + B + C) * size[test]) / (targetgb * GB)
  if (ratio > 0.90 && ratio  < 0.95) good=1; else good=0
  printf("gpu %s test %s size %d A %d B %d C %d targetgb %d ratio %2.3f good %d\n", gpu, test, size[test], A, B, C, targetgb, ratio, good)
} '







