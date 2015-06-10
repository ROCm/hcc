#include <amp.h>
#include <amp_math.h>
#include <amp_short_vectors.h>

using namespace Concurrency;
using namespace Concurrency::graphics;

extern Concurrency::array_view<float_2,1> *gbOutA;

extern void add(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut);

extern void sub(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut);
