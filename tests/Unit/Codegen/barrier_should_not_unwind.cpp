// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
using namespace Concurrency;

void
FwdPass0(const array_view<const int,1> &twiddles, Concurrency::tiled_index<64, 1> tidx) restrict(amp)
{
    tidx.barrier.wait();
}

int main()
{
    int num[1];
    const array_view<int,1>& twiddles = array_view<int, 1>(1, num);
    Concurrency::extent<2> grdExt( 64, 1 ); 
    Concurrency::tiled_extent< 64, 1> t_ext(grdExt);
    Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<64, 1> tidx) restrict(amp) {
        FwdPass0(twiddles,tidx);
        FwdPass0(twiddles,tidx);
    });

    return 0;
}
