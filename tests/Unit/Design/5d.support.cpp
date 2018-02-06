// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace Concurrency;

template<typename _type, int _rank>
bool test_array_rank(int extval = _rank)
{
    int *data = new int[_rank];
    for (int i = 0; i < _rank; i++)
        data[i] = extval;

    extent<_rank> e(data);
    array<_type, _rank> a1(e);

    parallel_for_each(e, [&](index<_rank> idx) restrict(amp) {
        a1[idx] = 1;
    });

    // is the rank correct
    if (a1.rank != _rank)
    {
        return false;
    }

    // verify data
    std::vector<_type> vdata = a1;
    for (unsigned int i = 0; i < e.size(); i++)
    {
        if (vdata[i] != 1)
            return false;
    }

    return true;
}

int main()
{
	int result = 1;

	result &= ((test_array_rank<int, 1>()));
	result &= ((test_array_rank<int, 5>()));
    
    return !result;
}
