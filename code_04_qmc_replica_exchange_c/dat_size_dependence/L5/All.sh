#!/bin/bash

# OpenBLAS
export OPENBLAS_NUM_THREADS=1
# MKL
#export MKL_NUM_THREADS=1
# BLIS
#export BLIS_NUM_THREADS=1
# macOS
#export VECLIB_MAXIMUM_THREADS=1

L=5

for i in \
`seq 1 16`
do
./qmc_kitaev --seed ${i} --L ${L} &
done
