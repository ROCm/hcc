#include <coordinate>
template <size_t N, template <size_t> class coord>
bool IsZero(const coord<N>& act)
{
    bool ret = true;
    for (int i = 0; i < N; i++)
        ret &= (act[i] == 0);
    ret &= (act.rank == N);
    return ret;
}
