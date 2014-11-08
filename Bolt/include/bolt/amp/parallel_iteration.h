/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

/*! \file bolt/amp/parallel_iteration.h
    \brief parallel_iteration just use a parallel_for for all devices
*/

#if !defined( BOLT_AMP_PARALLEL_ITER_H )
#define BOLT_AMP_PARALLEL_ITER_H

#pragma once

#include <amp.h>


//#include <bolt/combineable_queue.h>

//#define RESTRICT_AMP restrict(direct3d)


namespace bolt {

	/*! parallel_iteration1 : Slow implementation - just use a parallel_for for all devices.

	Init is the initial state for the Iteration type for every index.

	The user provides a functor that accepts two arguments:
	index<> - location in the extent to be processed.
	IterType : User-supplied type that tracks the state of each iteration.
	The functor is responsible for :
	* Updating the state so that it points to the next 'iteration'.  The next "iteration" could be the next cascade stage
	in a face-detection algorithm, or the next iteration of a loop.  It is very important to update to the next state to avoid
	an infinite loop.
	* Returns true if the Bolt runtime is to schedule the next iteration, or false if the runtime is to exit.
	*/
	template<typename IterType, typename Function>
	void parallel_iteration(concurrency::extent<1> ext, IterType Init, Function f)  {

		using namespace concurrency;

		parallel_for_each(extent<1>(ext), [=] (index<1> idx) mutable restrict(amp) {
			IterType iter(Init);

			bool keepGoing = true;

			while (keepGoing) {
				keepGoing = f(idx, iter);
			};
		});

	};

	/*!
	*/
	template<typename IterType, typename Function>
	void parallel_iteration_1(concurrency::extent<1> ext, IterType Init, Function f)  {

		using namespace concurrency;
		int splitPoint = 3;  // Iterations to run on the GPU:

		static const int maxQSz=1024;
		array<pair<index<1>, IterType> outQueue(maxQSz);
		array<int,1>  qPtr(1);

		parallel_for_each(extent<1>(ext), [=] (index<1> idx) mutable restrict(amp) {
			IterType iter(Init);
			int iterations = splitPoint;

			bool keepGoing = true;
			while (keepGoing && --iterations) {
				keepGoing = f(idx, iter);
			};
			if (keepGoing) {
				// Enqueue for processing below - save idx and state.
				//; Horrible performance due to contended atomic operations, need to combine enqueue operations.
				int lPtr = atomic_fetch_add(&qPtr[0], 1);
				// FIXME - bounds check
				if (lPtr < maxQSz) {
					outQueue[lPtr].first = idx;
					outQueue[lPtr].second = iter;
				}
			};
		});
	};
};

// TODO: Need versions to handle more extent dimensions??

// FIXME-TODO
// Add Doxygen-style comments:

#endif
