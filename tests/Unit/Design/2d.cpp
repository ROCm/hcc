    //_view RUN: %gtest_amp %s -o %t.out && %t.out

#include <hc.hpp>

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <vector>

using namespace hc;

class myVecAdd {
    array_view<int, 2> a_, b_, c_;
public:
    // CPU-side constructor. Written by the user
    myVecAdd(
        array_view<int, 2>& a, array_view<int, 2>& b, array_view<int, 2>& c)
        : a_(a), b_(b), c_(c)
    {}

    void operator()(index<2> idx) const [[hc]] { c_[idx] = a_[idx]+b_[idx]; }
    void operator()(tiled_index<2> idx) const [[hc]]
    {
        c_[idx] = a_[idx] + b_[idx];
    }
};

#define M 20
#define N 40

TEST(Design, Final)
{
    std::vector<int> vector_a(M * N), vector_b(M * N);

    for (int i = 0; i < M * N; i++) {
        vector_a[i] = 100.0f * rand() / RAND_MAX;
        vector_b[i] = 100.0f * rand() / RAND_MAX;
    }
    extent<2> e(M, N);
    array_view<int, 2> av(e, vector_a);
    EXPECT_EQ(vector_a[2], av(0, 2));
    array_view<int, 2> bv(e, vector_b);
    { // Test untiled version
        array_view<int, 2> c(e);
        myVecAdd mf(av, bv, c);
        parallel_for_each(e, mf);
        int error=0;
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                std::cout << "av[" <<i<<","<<j<<"] = "<<av(i,j)<<"\n";
                std::cout << "bv[" <<i<<","<<j<<"] = "<<bv(i,j)<<"\n";
                std::cout << "c[" <<i<<","<<j<<"] = "<<c(i,j)<<"\n";
                error += abs(c(i, j) - (av(i, j) + bv(i, j)));
            }
        }
        EXPECT_EQ(0, error);
    }
    {
        // Test tiled version
        array_view<int, 2> c(e);
        myVecAdd mf(av, bv, c);
        parallel_for_each(e.tile(4, 4), mf);
        int error=0;
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < N; j++) {
                std::cout << "av[" <<i<<","<<j<<"] = "<<av(i,j)<<"\n";
                std::cout << "bv[" <<i<<","<<j<<"] = "<<bv(i,j)<<"\n";
                std::cout << "c[" <<i<<","<<j<<"] = "<<c(i,j)<<"\n";
                error += abs(c(i, j) - (av(i, j) + bv(i, j)));
            }
        }
        EXPECT_EQ(0, error);
    }
}