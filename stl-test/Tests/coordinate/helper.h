#include <coordinate>
template <int N, template <int> class coord>
bool IsZero(const coord<N>& act)
{
    bool ret = true;
    for (int i = 0; i < N; i++)
        ret &= (act[i] == 0);
    ret &= (act.rank == N);
    return ret;
}
