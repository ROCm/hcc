#ifndef __INDEX_HELPERS__
#define __INDEX_HELPERS__

#include <amptest.h>
#include <vector>

using namespace Concurrency;
using namespace Concurrency::Test;
using std::vector;

/* Expects all index[i] to be set to 0 */
template<int Rank>
bool IsIndexSetToZero(const index<Rank> &actual)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
    {
        if(actual[i] != 0)
        {
            Log(LogType::Info, true) << "Fail: Incorrect index[" << i << "]. Actual: " << actual[i] << ".Expected: " << 0 << std::endl;
            passed = false;
        }
    }

    if (actual.rank != Rank)
    {
        Log(LogType::Info, true) << "Fail: Incorrect Rank. actual: " << actual.rank << ".Expected: " << Rank << std::endl;
        passed = false;
    }

    return passed;
}

/* Used to verify index created on device and returned as field, Expects index[i] to be set to 0 */
template<int Rank>
bool IsIndexSetToZero(vector<int> actual, int actualRank)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
    {
        if(actual[i] != 0)
        {
            Log(LogType::Info, true)<< "Fail: Incorrect actual[" << i << "]. Actual: " << actual[i] << ".Expected: " << 0 << std::endl;
            passed = false;
        }
    }

    if (actualRank != Rank)
    {
        Log(LogType::Info, true) << "Fail: Incorrect Rank. actual: " << actualRank << ".Expected: " << Rank << std::endl;
        passed = false;
    }

    return passed;
}

/* Expects all index[i] to be set to start + i */
template<int Rank>
bool IsIndexSetToSequence(const index<Rank> &actual, int start = 0)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
    {
        if(actual[i] != start + i)
        {
            Log(LogType::Info, true)<< "Fail: Incorrect index[" << i << "]. Actual: " << actual[i] << ".Expected: " << start + i << std::endl;
            passed = false;
        }
    }

    if (actual.rank != Rank)
    {
        Log(LogType::Info, true)<< "Fail: Incorrect Rank. actual: " << actual.rank << ".Expected: " << Rank << std::endl;
        passed = false;
    }

    return passed;
}

/* Used to verify index created on device and returned as field. Expects index[i]) to be set to start + i */
template<int Rank>
bool IsIndexSetToSequence(vector<int> actual, int actualRank, int start = 0)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
    {
        if(actual[i] != start + i)
        {
            Log(LogType::Info, true)<< "Fail: Incorrect actual[" << i << "]. Actual: " << actual[i] << ".Expected: " << start + i << std::endl;
            passed = false;
        }
    }

    if (actualRank != Rank)
    {
        Log(LogType::Info, true)<< "Fail: Incorrect Rank. actual: " << actualRank << ".Expected: " << Rank << std::endl;
        passed = false;
    }

    return passed;
}

#endif /* __INDEX_HELPERS__ */
