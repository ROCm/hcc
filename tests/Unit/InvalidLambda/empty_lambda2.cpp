// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
int main()
{
    int gpu_result;
	concurrency::array_view<int> gpu_resultsv(1, &gpu_result);
    gpu_resultsv.discard_data();
    auto fun = [&]() restrict(cpu,amp) { return 0; };
    concurrency::parallel_for_each(gpu_resultsv.get_extent(), [=] (concurrency::index<1> idx) restrict (amp) { gpu_resultsv[idx] = fun(); });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Wrong scripts to cause "__ATTRIBUTE_CTOR__" without any identifier in kernel.cl
//
#if 0
// R1UN: %amp_device -D__KALMAR_ACCELERATOR__ %s -m32 -emit-llvm -c -S -O3 -o %t.ll && mkdir -p %t
// R1UN: %llc -march=c -o %t/kernel_.cl < %t.ll
// R1UN: cat %opencl_math_dir/opencl_math.cl %t/kernel_.cl > %t/kernel.cl
// R1UN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// R1UN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////
