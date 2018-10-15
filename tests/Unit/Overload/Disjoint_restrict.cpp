// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
#include <hc_math.hpp>
#include <iostream>
using namespace hc;

int test() [[cpu, hc]]
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
	runall_result() [[cpu, hc]]
		: _exit_code(0)
	{}

	runall_result(int result) [[cpu, hc]]
		: _exit_code(result)
	{
		verify_exit_code();
	}


private:
	int _exit_code;

	void verify_exit_code() [[cpu]];
	void verify_exit_code() [[hc]] {}
};

void runall_result::verify_exit_code() [[cpu]]
{
      if(_exit_code != 0)
      {
        throw std::invalid_argument(".");
      }
}

int main()
{
	runall_result gpu_result;
	hc::array_view<runall_result> gpu_resultsv(1, &gpu_result);

	hc::parallel_for_each(gpu_resultsv.get_extent(), [=](hc::index<1> idx) [[hc]]
	{
		gpu_resultsv[idx] = test();
	});
}
