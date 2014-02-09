// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && %embed_kernel kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out
#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;
using std::vector;

/* --- All index variables are declared as Index<RANK> {0, 1, 2 ..} for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */

// Pass index by value to invoke copy constructor
bool func(index<RANK> idx)
{
   return IsIndexSetToSequence<RANK>(idx);
}

bool CopyConstructWithIndexOnHost()
{
    index<RANK> idx(0, 1, 2);    
    return func(idx);
}

/*---------------- Test on Device ---------------- */

// idx is copy constructed between vector functions
void k1(array<int, 1>& C, array<int, 1>& D, const index<RANK>& idx) restrict(amp,cpu)
{
    for(int i = 0; i < RANK;i++)
    {
        C(i) = idx[i];
    }

    D(0) = idx.rank;
}

// idx is copy constructed in the kernel function
void kernel(array<int, 1>& A, array<int, 1>& B, array<int, 1>& C, array<int, 1>& D, const index<RANK>& idx) restrict(amp,cpu)
{
    for(int i = 0; i < RANK;i++)
    {
        A(i) = idx[i];
    }

    B(0) = idx.rank;

    k1(C, D, idx);
}

int CopyConstructWithIndexOnDevice()
{
	int result = 1;

    index<RANK> idxparam(0, 1, 2);

    vector<int> vA(RANK), vB(1), vC(RANK), vD(1);
    array<int, 1> A((extent<1>(RANK))), B(extent<1>(1)), C((extent<1>(RANK))), D(extent<1>(1));

    extent<1> ex(1);
    parallel_for_each(ex, [&idxparam, &A, &B, &C, &D](index<1> idx) restrict(amp,cpu) {
        kernel(A, B, C, D, idxparam);
    });

    vA = A;
    vB = B;
    vC = C;
    vD = D;

    result &= (IsIndexSetToSequence<RANK>(vA, vB[0]));
    result &= (IsIndexSetToSequence<RANK>(vC, vD[0]));
    result &= (vB[0] == vD[0]);
   
    return result;
}


int main() 
{
	int result = 1;
    result &= (CopyConstructWithIndexOnHost());
    result &= (CopyConstructWithIndexOnDevice());
	return !result;
}
