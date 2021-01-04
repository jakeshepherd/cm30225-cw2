#!/bin/bash

mpicc cw2.c -o cw2
mpirun -np 2 cw2 6 0.01 input-data/input-data-6.txt
