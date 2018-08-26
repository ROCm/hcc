#include <hc.hpp>
#include <hc_math.hpp>
#include <amp_short_vectors.h>

using namespace hc;
using namespace hc::graphics;

extern hc::array_view<float_2,1> *gbOutA;

extern void add(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut);

extern void sub(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut);
