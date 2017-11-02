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
