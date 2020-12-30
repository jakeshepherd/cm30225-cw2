#!/bin/bash

mpicc cw2.c -o cw2
mpirun -np 1 cw2 0.01 input-data/input-data-200.txt
