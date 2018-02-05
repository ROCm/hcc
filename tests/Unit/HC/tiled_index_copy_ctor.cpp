// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>

using namespace hc;

[[hc]] inline void driver(hc::array<int> &a,  hc::tiled_index<3> i)
{
  a[i.global[0]] = 1;
}


int main (void)
{
  std::vector<int> vec(64);
  hc::array<int> hca(vec.size());

  // now use a PFE to operate on the vector.

  extent< 3 > e(64,1,1);
  tiled_extent< 3 > te = e.tile(64,1,1);

  parallel_for_each(te,
                    [=,&hca]
                    (tiled_index<3> idx) [[hc]]
                    {
                    //  tiled_index<3> idx_copy = idx;
                    driver(hca,idx);
                    });
  copy(hca,vec.data());

  for(int c: vec) {
    if (c != 1) {
      return 1;
    }
  }

  return 0;
}

