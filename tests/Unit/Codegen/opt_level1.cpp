// RUN: %cxxamp -O1 %s -o %t.out 2>&1 | %FileCheck %s
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <vector>
#include <amp.h>
using namespace Concurrency;

#define N 10

void randomInit(std::vector<float>& data, int n)
{
    for (int i = 0; i < n; ++i)
	data[i] = rand() / (float)RAND_MAX;
}

void vectorAdd_by_array(const std::vector<float>& vecA, const std::vector<float>& vecB){
   array<float> A(N, vecA.begin());
   array<float> B(N, vecB.begin());
   array<float> C(N);
   array_view<float> av(A);
   array_view<float> bv(B);
   array_view<float> cv(C);
   extent<1> e(N);

   parallel_for_each(e,
         [=](index<1> idx) restrict(amp) { cv[idx] = av[idx] + bv[idx]; }); 
}

int main(void)
{
    std::vector<float> vecA(N);
    std::vector<float> vecB(N);
    randomInit(vecA, N);
    randomInit(vecB, N);

    vectorAdd_by_array(vecA, vecB);

    return 0;
}

// CHECK: clang-{{[0-9]}}.{{[0-9]}}: warning: -O1 is ignored in GPU compilation path
