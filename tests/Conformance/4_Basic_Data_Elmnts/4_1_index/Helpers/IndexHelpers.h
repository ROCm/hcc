#ifndef __INDEX_HELPERS__
#define __INDEX_HELPERS__

#include <amp.h>
#include <vector>

using namespace Concurrency;
using std::vector;

/* Expects all index[i] to be set to 0 */
template<int Rank>
bool IsIndexSetToZero(const index<Rank> &actual)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
        if(actual[i] != 0)
            passed = false;

    if (actual.rank != Rank)
        passed = false;

    return passed;
}

/* Used to verify index created on device and returned as field, Expects index[i] to be set to 0 */
template<int Rank>
bool IsIndexSetToZero(vector<int> actual, int actualRank)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
        if(actual[i] != 0)
            passed = false;

    if (actualRank != Rank)
        passed = false;

    return passed;
}

/* Expects all index[i] to be set to start + i */
template<int Rank>
bool IsIndexSetToSequence(const index<Rank> &actual, int start = 0)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
        if(actual[i] != start + i)
            passed = false;

    if (actual.rank != Rank)
        passed = false;

    return passed;
}

/* Used to verify index created on device and returned as field. Expects index[i]) to be set to start + i */
template<int Rank>
bool IsIndexSetToSequence(vector<int> actual, int actualRank, int start = 0)
{
    bool passed = true;
    for(int i = 0; i < Rank; i++)
        if(actual[i] != start + i)
            passed = false;

    if (actualRank != Rank)
        passed = false;

    return passed;
}

#endif /* __INDEX_HELPERS__ */
