// RUN: %amp_device -D__GPU__ %s -m32 -emit-llvm -c -S -O2 -o %t.ll && mkdir -p %t
// RUN: %clamp-device %t.ll %t/kernel.cl
// RUN: pushd %t && objcopy -B i386:x86-64 -I binary -O elf64-x86-64 kernel.cl %t/kernel.o && popd
// RUN: %cxxamp %link %t/kernel.o %s -o %t.out && %t.out

#include "../../../Helpers/IndexHelpers.h"
#include <amp.h>
using namespace Concurrency;
using std::vector;


/* --- All index variables are declared as Index<RANK> {0, 1, 2..} for ease of verification --- */
const int RANK = 3;

/*---------------- Test on Host ------------------ */
bool CopyConstructWithIndexOnHost()
{

    index<RANK> idx1(0, 1, 2);
    index<RANK> idx2(idx1);   // copy construct
    
    return IsIndexSetToSequence<RANK>(idx2);
}

/*---------------- Test on Device ---------------- */
/* A returns the components of the index, B returns the Rank */
void kernelIndex(array<int, 1>& A, array<int, 1>& B) restrict(amp,cpu)
{
    index<RANK> index1(0, 1, 2);
    index<RANK> index2(index1);   // copy construct

    for(int i = 0; i < RANK;i++)
    {
        A(i) = index2[i];
    }

    B(0) = index2.rank;
}

bool CopyConstructWithIndexOnDevice()
{
    vector<int> resultsA(RANK), resultsB(1);
    array<int, 1> A((extent<1>(RANK))), B(extent<1>(1));

    extent<1> ex(1);
    parallel_for_each(ex, [&](index<1> idx) restrict(amp,cpu) {
        kernelIndex(A, B);
    });

    resultsA = A;
    resultsB = B;

    return IsIndexSetToSequence<RANK>(resultsA, resultsB[0]);
}


/*--------------------- Main -------------------- */
int main() 
{
    int result = 1;

    // Test on host
    result &= (CopyConstructWithIndexOnHost());

    // Test on device
    result &= (CopyConstructWithIndexOnDevice());
    return !result;
}
