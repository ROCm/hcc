#include <iomanip>

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"


#define failed(...) \
    printf ("%serror: ", KRED);\
    printf (__VA_ARGS__);\
    printf ("\n");\
    printf ("error: TEST FAILED\n%s", KNRM );\
    abort();

#define HostToDeviceCopyTest 0x1
#define DeviceToDeviceCopyTest 0x2
#define DeviceToHostCopyTest 0x4
#define HostToDeviceAsyncCopyTest 0x8	
#define DeviceToHostAsyncCopyTest 0x10

template <typename T>
void freeArraysForHost(T *A_h, T *B_h, T *C_h, bool usePinnedHost)
{
    if (usePinnedHost) {
        if (A_h) {
            hc::am_free(A_h);
        }
        if (B_h) {
            hc::am_free(B_h);
        }
        if (C_h) {
            hc::am_free(C_h);
        }
    } else {
        if (A_h) {
            free (A_h);
        }
        if (B_h) {
            free (B_h);
        }
        if (C_h) {
            free (C_h);
        }
    }

}

template <typename T>
void freeArrays(T *A_d, T *B_d, T *C_d,
                T *A_h, T *B_h, T *C_h, bool usePinnedHost) 
{
    if (A_d) {
        hc::am_free(A_d) ;
    }
    if (B_d) {
        hc::am_free(B_d) ;
    }
    if (C_d) {
        hc::am_free(C_d) ;
    }

    freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}


// Assumes C_h contains vector add of A_h + B_h
// Calls the test "failed" macro if a mismatch is detected.
template <typename T>
void checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch=true)
{
    size_t  mismatchCount = 0;
    size_t  firstMismatch = 0;
    size_t  mismatchesToPrint = 10;
    for (size_t i=0; i<N; i++) {
        T expected = A_h[i] + B_h[i];
        if (result_H[i] != expected) {
            if (mismatchCount == 0) {
                firstMismatch = i;
            }
            mismatchCount++;
            if ((mismatchCount <= mismatchesToPrint) && expectMatch) {
                std::cout << std::fixed << std::setprecision(32);
                std::cout << "At " << i << std::endl;
                std::cout << "  Computed:" << result_H[i]  << std::endl;
                std::cout << "  Expected:" << expected << std::endl;
            }
        }
    }

    if (expectMatch) {
        if (mismatchCount) {
            failed("%zu mismatches ; first at index:%zu\n", mismatchCount, firstMismatch);
        }
    } else {
        if (mismatchCount == 0) {
            failed("expected mismatches but did not detect any!");
        }
    }

}
