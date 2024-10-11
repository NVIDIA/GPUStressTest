#! /bin/bash

input=${1:-gst.out}

cat $input | awk ' 
BEGIN {
  GB=1073741824.0
  size["INT8"]=1
  size["FP8"]=1
  size["FP16"]=2
  size["TF32"]=4
  size["FP64"]=8
  size["FP32"]=4
  size["C32"]=4
}
/Initilizing/ {gpu=$2} 
/memgb:/ {targetgb=$5}
/STARTING TEST/ {test=$5}
/FINAL/ {A=$6 + 0.0; B=$8i + 0.0; C=$10 + 0.0}
/PASSED/ {
  ratio=((((A + B + C) * size[test]) / GB) / targetgb)
  if (ratio > 0.80 && ratio  < 0.95) good=1; else good=0
  printf("gpu %s test %s size %ld A %ld B %ld C %ld targetgb %ld ratio %2.3f good %d\n", gpu, test, size[test], A, B, C, targetgb, ratio, good)
} '







