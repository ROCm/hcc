// RUN: %cxxamp -DFILE_1 -c %s -o %t.1.o
// RUN: %cxxamp -DFILE_2 -c %s -o %t.2.o
// RUN: %cxxamp %t.1.o %t.2.o -o %t.out

// this test try to link 2 files which use C++AMP short vectors API in one
// executable.  so it tests if short vector APIs would violate ODR rule
#ifdef FILE_1
#include "amp_short_vectors_2files.h"

void add(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut) 
{
  Concurrency::extent<2> grdExt(64, 1);
  Concurrency::tiled_extent<64, 1> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<64,1> tidx) restrict(amp)
  {

	unsigned int me = tidx.global[0];

	if(me < 64)
	{
	  gbOut[me].set_x(gbIn[me]);
	  gbOut[me].set_y(gbIn[me]);
	}

   });
}

int main()
{
    float *gbIn = (float*) calloc(64, sizeof(float));
    float_2 *gbOut = (float_2*) calloc(64, sizeof(float_2));
    
    for(int i = 0; i< 64;i++)
    {
      gbIn[i] = i + 1;
      gbOut[i].set_x(i + 1);
      gbOut[i].set_y(i + 1);
    }
    
    const Concurrency::array_view<float, 1> gbInA(64, gbIn);
    const Concurrency::array_view<float_2, 1> gbOutAB(64, gbOut);

    add(gbInA, gbOutAB); 

    gbOutAB.synchronize();
    
    sub(gbInA, gbOutAB); 

    gbOutAB.synchronize();
    
    /* Print Output */
    /*for (int i = 0; i < 64; i++)
        std::cout<<" gbOutA["<<i<<"] "<<gbOutA[i].x << " y "<< gbOutA[i].y<<std::endl;*/

}

#else
#include "amp_short_vectors_2files.h"

concurrency::array_view<float_2,1> *gbOutA;

void sub(const array_view<float,1> &gbIn,const array_view<float_2,1> &gbOut) 
{
  Concurrency::extent<2> grdExt(64, 1);
  Concurrency::tiled_extent<64, 1> t_ext(grdExt);

  Concurrency::parallel_for_each(t_ext, [=] (Concurrency::tiled_index<64,1> tidx) restrict(amp)
  {

	unsigned int me = tidx.global[0];

	if(me < 64)
	{
	  gbOut[me].set_x(gbIn[me]);
	  gbOut[me].set_y(gbIn[me]);
	}

   });
}
#endif
