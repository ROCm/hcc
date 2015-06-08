// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace concurrency;

#define TEST_CPU
#define TEST_ELIDED
#define TEST_GPU
#define TEST_BOTH

int callee() restrict(amp)
{
    return 1;
}

int callee()
{
    return 2;
}

#ifdef TEST_CPU
bool CPU_Func()
{
    // in CPU path, return 1; in GPU path, return 0

    if (callee() != 2)
    {
        return false;
    }

    return true;
}
#endif

#ifdef TEST_ELIDED
bool Elided_Func()
{
    if (callee() != 2)
    {
        return false;
    }

    return true;
}
#endif

#ifdef TEST_GPU
bool AMP_Func() restrict(amp)
{
    if (callee() != 1)
    {
        return false;
    }

    return true;
}
#endif

#ifdef TEST_BOTH
bool BOTH_CPU_AND_AMP() restrict(cpu,amp)
{
#if __KALMAR_ACCELERATOR__
    if (callee() != 1)
#else
    if (callee() != 2)
#endif
    {
        return false;
    }

    return true;
}
#endif

int main(int argc, char **argv)
{
    int flag;
#ifdef TEST_CPU
    flag = CPU_Func()? 0 : 1;
    if(flag) { printf("CPU_Func Error! exit!\n"); exit(1);}
#endif
#ifdef TEST_ELIDED
    flag = Elided_Func()?0:1;
    if(flag) { printf("Elided_Func Error! exit!\n"); exit(1);}
#endif
#ifdef TEST_GPU
    // directly called is not allowed, we use pfe
    {
      int result;
      concurrency::array_view<int> gpu_resultsv(1, &result);
      concurrency::parallel_for_each(gpu_resultsv.get_extent(), [=](concurrency::index<1> idx) restrict(amp)
      {
        gpu_resultsv[idx] = AMP_Func();
      });
    
       if(gpu_resultsv[0] == 0) { printf("AMP_Func Error! exit!\n"); exit(1);}
     }
#endif

#ifdef TEST_BOTH
    {
      int result;
      concurrency::array_view<int> gpu_resultsv(1, &result);
      concurrency::parallel_for_each(gpu_resultsv.get_extent(), [=](concurrency::index<1> idx) restrict(amp,cpu)
      {
        gpu_resultsv[idx] = BOTH_CPU_AND_AMP();
      });
    
       if(gpu_resultsv[0] == 0) { printf("BOTH_CPU_AND_AMP Error! exit!\n"); exit(1);}
     }
#endif

    return 0;
}
