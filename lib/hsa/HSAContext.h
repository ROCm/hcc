// University of Illinois/NCSA
// Open Source License
// 
// Copyright (c) 2013, Advanced Micro Devices, Inc.
// All rights reserved.
// 
// Developed by:
// 
//     Runtimes Team
// 
//     Advanced Micro Devices, Inc
// 
//     www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal with
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// 
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimers.
// 
//     * Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimers in the
//       documentation and/or other materials provided with the distribution.
// 
//     * Neither the names of the LLVM Team, University of Illinois at
//       Urbana-Champaign, nor the names of its contributors may be used to
//       endorse or promote products derived from this Software without specific
//       prior written permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
// SOFTWARE.
//===----------------------------------------------------------------------===//

#ifndef HSACONTEXT_H
#define HSACONTEXT_H
#include <future>
#include <hsa.h>

// Abstract interface to an HSA Implementation
class HSAContext{
public:
        class Dispatch {
        public:
		// various methods for setting different types of args into the arg stack
		virtual hsa_status_t pushFloatArg(float) = 0;
		virtual hsa_status_t pushIntArg(int) = 0;
		virtual hsa_status_t pushBooleanArg(unsigned char) = 0;
		virtual hsa_status_t pushByteArg(char) = 0;
		virtual hsa_status_t pushLongArg(long) = 0;
		virtual hsa_status_t pushDoubleArg(double) = 0;
		virtual hsa_status_t pushPointerArg(void *addr) = 0;
		virtual hsa_status_t clearArgs() = 0;

		// setting number of dimensions and sizes of each
		virtual hsa_status_t setLaunchAttributes(int dims, size_t *globalDims, size_t *localDims) = 0;

		// run a kernel and wait until complete
		virtual hsa_status_t dispatchKernelWaitComplete() = 0;

                // dispatch a kernel asynchronously
                virtual hsa_status_t dispatchKernel() = 0;

                // wait for the kernel to finish execution
                virtual hsa_status_t waitComplete() = 0;

                // dispatch a kernel asynchronously and get a future object
                virtual std::shared_future<void>* dispatchKernelAndGetFuture() = 0;

                // destructor
                virtual ~Dispatch() {}
        };

	class Kernel {
	};

	// create a kernel object from the specified HSAIL text source and entrypoint
	virtual Kernel* createKernel(const char *source, int size, const char *entryName) = 0;

        // create a kernel dispatch object from the specified kernel
        virtual Dispatch* createDispatch(const Kernel *kernel) = 0;

	// dispose of an environment including all programs
	virtual hsa_status_t dispose() = 0;

	virtual hsa_status_t registerArrayMemory(void *addr, int lengthInBytes) = 0;

	static HSAContext* Create();

private:
    static HSAContext* m_pContext;
};

#endif // HSACONTEXT_H
