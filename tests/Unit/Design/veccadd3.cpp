// RUN: %cxxamp %s -o %t.out && %t.out
#include <amp.h>
#include <iostream>

using namespace concurrency;

void vecAdd(float* A, float* B, float* C, int n)

{
    accelerator acc;
    accelerator_view view(acc.get_default_view());
    array<float,1> AA(n,view), BA(n,view);
    array<float,1> CA(n,view);
    copy(A,AA);
    copy(B,BA);	
    parallel_for_each(view, CA.get_extent(), 
            [&AA,&BA,&CA](index<1> i) restrict(amp) {
            CA[i] = AA[i] + BA[i];
    });
    copy(CA,C);
}

bool verify(float *A, float *B, float *C, int n) {

    const float relativeTolerance = 1e-6;

    for(int i = 0; i < n; i++) {
        float sum = A[i] + B[i];
        float relativeError = (sum - C[i])/sum;
        if (relativeError > relativeTolerance
                || relativeError < -relativeTolerance) {
            std::cout << "\nTEST FAILED\n\n";
            return true;
        }
    }
    std::cout << "\nTEST PASSED\n\n";
    return false;
}

int main(int argc, char**argv) {

    // Initialize host variables ----------------------------------------------
    std::vector<accelerator> accs = accelerator::get_all();
    std::wcout << "Size: " << accs.size() << std::endl;                       
    if(accs.size() == 0) {
      std::wcout << "There is no acclerator!\n";
      // Since this case is to test on GPU device, skip if there is CPU only
      return 0;
    }
    assert(accs.size() && "Number of Accelerators == 0!");
    std::for_each(accs.begin(), accs.end(), [&] (accelerator acc) 
            {  
            std::wcout << "New accelerator: " << acc.get_description() << std::endl;
            std::wcout << "device_path = " << acc.get_device_path() << std::endl;
            std::wcout << "version = " << (acc.get_version() >> 16) << '.' << (acc.get_version() & 0xFFFF) << std::endl;
            std::wcout << "dedicated_memory = " << acc.get_dedicated_memory() << " KB" << std::endl;
            std::wcout << "doubles = " << ((acc.get_supports_double_precision()) ? "true" : "false") << std::endl;
            std::wcout << "limited_doubles = " << ((acc.get_supports_limited_double_precision()) ? "true" : "false") << std::endl;
            std::wcout << "has_display = " << ((acc.get_has_display()) ? "true" : "false") << std::endl;
            std::wcout << "is_emulated = " << ((acc.get_is_emulated()) ? "true" : "false") << std::endl;
            std::wcout << "is_debug = " << ((acc.get_is_debug()) ? "true" : "false") << std::endl;
            std::cout << std::endl;
            });

    std::cout << "\nSetting up the problem..." << std::endl;

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        std::cout <<
            "\n    Invalid input parameters!"
            "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
            "\n    Usage: ./vecadd <m>           # Vector of size m is used"
            "\n\n";
        exit(0);
    }

    float* A = new float [n];
    for (unsigned int i=0; i < n; i++) { A[i] = (rand()%100)/100.00; }

    float* B = new float [n];
    for (unsigned int i=0; i < n; i++) { B[i] = (rand()%100)/100.00; }

    float* C = new float [n];

    std::cout << "    Vector size = " << n << std::endl;

    // Launch Kernel ----------------------------------------------------------

    vecAdd(A, B, C, n);

    // Verify correctness -----------------------------------------------------

    std::cout << "Verifying results..." << std::endl;

    bool failed = verify(A, B, C, n);

    // Free memory ------------------------------------------------------------

    delete [] A;
    delete [] B;
    delete [] C;

    return failed? -1:0;
}



