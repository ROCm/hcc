// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <amp_math.h>
#include <iostream>
using namespace concurrency;

int test() restrict(cpu,amp)
{
    int data[] = {1};
    for (int i = 0; i < 1; i++)
    {
        if (data[i] != i + 1)
        {
            return 1;
        }
    }

    return 0;
}

struct runall_result
{
	runall_result() restrict(cpu,amp)
		: _exit_code(0)
	{}

	runall_result(int result) restrict(cpu,amp)
		: _exit_code(result)
	{
		verify_exit_code();
	}


private:
	int _exit_code;

	void verify_exit_code() restrict(cpu);
	void verify_exit_code() restrict(amp) {}
};

void runall_result::verify_exit_code() restrict(cpu)
{
      if(_exit_code != 0)
      {
        throw std::invalid_argument(".");
      }
}

int main()
{
	runall_result gpu_result;
	concurrency::array_view<runall_result> gpu_resultsv(1, &gpu_result);

	concurrency::parallel_for_each(gpu_resultsv.get_extent(), [=](concurrency::index<1> idx) restrict(amp)
	{
		gpu_resultsv[idx] = test();
	});
}
