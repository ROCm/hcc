// XFAIL: Linux
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

#include <iostream>

using namespace concurrency;

#define MAXSIZE (64) 

// simple test of c++AMP to discover GPU workgroup sizes
// 3D version

//think of M as number of rows, N and number of columns, like fortran
template<int TM,int TN, int TL>
void at(int *di1,int *di2,int *di3,int *di4,int *di5,int *di6,int nx,int ny,int nz)
{
      extent<3> e(nx,ny,nz);
//total number of threads in a tile can't exceed 1024
// up to 1,32,32 supported
//max local x extent is 2 for 3D case
//max local y extent is 32
//max local z extent is 32
      parallel_for_each(e.tile<TM,TN,TL>(),
      [=]
      (tiled_index<TM,TN,TL> idx) restrict(amp)
      {
      int lid0 = idx.local[0];
      int lid1 = idx.local[1];
      int lid2 = idx.local[2];
      int gid0 = idx.global[0];
      int gid1 = idx.global[1];
      int gid2 = idx.global[2];
      int ny2 = ny*ny;
      int gidx = gid2*ny2+gid1*nx+gid0;
         di1[gidx] = gid0;
         di2[gidx] = gid1;
         di3[gidx] = gid2;
         di4[gidx] = lid0;
         di5[gidx] = lid1;
         di6[gidx] = lid2;
      });
}
#define MS  MAXSIZE
#define SQ  MAXSIZE*MAXSIZE
#define CB  MAXSIZE*MAXSIZE*MAXSIZE

int di1[CB],di2[CB];
int di3[CB],di4[CB];
int di5[CB],di6[CB];

template<size_t Z, size_t Y, size_t X, size_t MAX_Z, size_t MAX_Y, size_t MAX_X, size_t MAX_ALL>
bool test() {
  bool ret = true;

  // launch kernel to keep track of all global IDs and tile IDs
  at<Z,Y,X>(di1,di2,di3,di4,di5,di6,MS,MS,MS);

#if 0
  printf("SIZE %d, CUBE %d\n",MS,CB);
  printf("  x   y  z    g0  g1  g2    l0  l1  l2\n");
  for(int k=0;k<MAXSIZE;k++)
   for(int j=0;j<MAXSIZE;j++)
     for(int i=0;i<MAXSIZE;i++) {
         printf("%3d %3d %3d   %3d %3d %3d   %3d %3d %3d\n",i,j,k,
           di1[i+j*MS+k*SQ],di2[i+j*MS+k*SQ],
           di3[i+j*MS+k*SQ],di4[i+j*MS+k*SQ],
           di5[i+j*MS+k*SQ],di6[i+j*MS+k*SQ]);
#endif

  int maxX = 0;
  int maxY = 0;
  int maxZ = 0;

  // walkthrough one specified tile and figure out the maximum workgroup
  // size for each dimension
  for(int k=0;k<MAXSIZE;k++)
   for(int j=0;j<MAXSIZE;j++)
     for(int i=0;i<MAXSIZE;i++) {
       maxZ = std::max(maxZ, di4[i+j*MS+k*SQ]);
       maxY = std::max(maxY, di5[i+j*MS+k*SQ]);
       maxX = std::max(maxX, di6[i+j*MS+k*SQ]);
     }

  // thread local id are indexed from 0
  // so the maximum size  is one plus the maximum workgroup ID found
  ++maxX; ++maxY; ++maxZ;

#if 0
  std::cout << maxX << " " << maxY << " " << maxZ << "\n";
#endif

  // check if each workgroup dimension is within limit
  if (maxX > MAX_X) ret = false;
  if (maxY > MAX_Y) ret = false;
  if (maxZ > MAX_Z) ret = false;

  // check if the workgroup size is within limit
  if (maxX * maxY * maxZ > MAX_ALL) {
    ret = false;
  }

  return ret;
}

int main() {
  bool ret = true;

  ret &= test<4, 8, 8, 256, 256, 256, 256>();   // no truncation would take place
  ret &= test<8, 8, 8, 256, 256, 256, 256>();   // will be truncated to 4, 8, 8
  ret &= test<64, 16, 1, 256, 256, 256, 256>(); // will be truncated to 32, 8, 1
  ret &= test<64, 4, 4, 256, 256, 256, 256>();  // will be truncated to 32, 2, 4

  return !(ret == true);
}

