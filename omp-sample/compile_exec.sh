#!/bin/bash

clang++ -fopenmp md_openmp.cpp -o md_openmp
clang -fopenmp omp_hello.c -o omp_hello
clang -fopenmp omp_mm.c -o omp_mm
clang -fopenmp omp_orphan.c -o omp_orphan
clang -fopenmp omp_reduction.c -o omp_reduction
clang -fopenmp omp_workshare1.c -o omp_workshare1

./md_openmp && ./omp_hello && ./omp_mm && ./omp_orphan && ./omp_reduction && ./omp_workshare1
