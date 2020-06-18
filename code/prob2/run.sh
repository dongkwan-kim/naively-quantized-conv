#!/usr/bin/env bash
NUM=${1:-1}
QUANT=${2:-0}
CONST=${3:-0}
make clean
make
echo "./convolution ../../group2/${NUM}/input_tensor.bin ../../group2/${NUM}/kernel_tensor.bin"
./convolution ../../group2/${NUM}/input_tensor.bin ../../group2/${NUM}/kernel_tensor.bin ${QUANT} ${CONST}