// RUN: %cxxamp %s -o %t.out && %t.out
#include <hc.hpp>
using namespace hc;

void
FwdPass0(const array_view<const int,1> &twiddles, hc::tiled_index<2> tidx) [[hc]]
{
    tidx.barrier.wait();
}

int main()
{
    int num[1];
    const array_view<int,1>& twiddles = array_view<int, 1>(1, num);
    hc::extent<2> grdExt( 64, 1 ); 
    hc::tiled_extent<2> t_ext(grdExt.tile(64, 1));
    hc::parallel_for_each(t_ext, [=] (hc::tiled_index<2> tidx) [[hc]] {
        FwdPass0(twiddles,tidx);
        FwdPass0(twiddles,tidx);
    });

    return 0;
}
