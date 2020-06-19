#!/usr/bin/env bash
NUM=${1:-1}
QUANT=${2:-"FP32"}
make clean
make
echo "./convolution ../../group2/${NUM}/input_tensor.bin ../../group2/${NUM}/kernel_tensor.bin ${QUANT}"
./convolution ../../group2/${NUM}/input_tensor.bin ../../group2/${NUM}/kernel_tensor.bin ${QUANT}