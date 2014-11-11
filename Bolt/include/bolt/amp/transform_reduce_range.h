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

/*! \file bolt/amp/transform_reduce_range.h
    \brief  Takes a start index and extent as the range to iterate.
*/

#if !defined( BOLT_AMP_TRANSFORM_REDUCE_RANGE_H )
#define BOLT_AMP_TRANSFORM_REDUCE_RANGE_H

#pragma once

#include <bolt/transform_reduce.h>


#ifdef BOLT_POOL_ALLOC
#include <bolt/pool_alloc.h>
// hacky global var, move to bolt.cpp?
bolt::ArrayPool<HessianState> arrayPool;
#endif

namespace bolt {
	namespace amp {

#define VW 1
#define BARRIER(W)  // FIXME - placeholder for future barrier insertions

#define REDUCE_STEP(_IDX, _W) \
	if (_IDX < _W) tiled_data[_IDX] = reduce_op(tiled_data[_IDX], tiled_data[_IDX+_W]); \
	BARRIER(_W)


	//=======================
	// This version takes a start index and extent as the range to iterate.
	// The tranform_op is called with topLeft, bottomRight, and stride for each section that to be processed by
	// the function.  Function must do iteration over the specified range with specified stride, and also reduce results.
	// May avoid copies on some compilers and deliver higher performance than the cleaner transform_reduce function,
	// where transform op only takes a single index point.
	// Useful for indexing over all points in an image or array
	template<typename outputT, int Rank, typename UnaryFunction, typename BinaryFunction>
	outputT transform_reduce_range(concurrency::accelerator_view av,
		concurrency::index<Rank> origin, concurrency::extent<Rank> ext,
		UnaryFunction transform_op,
		outputT init,  BinaryFunction reduce_op)
	{
		using namespace concurrency;

		int wgPerComputeUnit = p_wgPerComputeUnit; // remove me.
		int computeUnits     = p_computeUnits;
		int resultCnt = computeUnits * wgPerComputeUnit;

		// FIXME: implement a more clever algorithm for setting the shape of the calculation.
		int globalH = wgPerComputeUnit * localH;
		int globalW = computeUnits * localW;

		globalH = (ext[0] < globalH) ? ext[0] : globalH; //FIXME, this is not a multiple of localSize.
		globalW = (ext[1] < globalW) ? ext[1] : globalW;


		extent<2> launchExt(globalH, globalW);
#ifdef BOLT_POOL_ALLOC
		bolt::ArrayPool<outputT>::PoolEntry &entry = arrayPool.alloc(av, resultCnt);
		array<outputT,1> &results1 = *(entry._dBuffer);
#else
		array<outputT,1> results1(resultCnt, av);  // Output after reducing through LDS.
#endif
		index<2> bottomRight(origin[0]+ext[0], origin[1]+ext[1]);

		// FIXME - support actual BARRIER operations.
		// FIXME - support checks on local memory usage
		// FIXME - reduce size of work for small problems.
		concurrency::parallel_for_each(av,  launchExt.tile<localH, localW>(), [=,&results1](concurrency::tiled_index<localH, localW> idx) mutable restrict(amp)
		{
			tile_static outputT tiled_data[waveSize];

#if 1
			init = reduce_op(init, transform_op(index<Rank>(origin[0]+idx.global[0], origin[1]+idx.global[1]),  //top/left
				bottomRight,  // bottomRight
				launchExt));  //stride
#endif

#if 0
			for (int y=origin[0]+idx.global[0]; y<origin[0]+ext[0]; y+=launchExt[0]) {
				for (int x=origin[1]+idx.global[1]*VW; x<origin[1]+ext[1]; x+=launchExt[1]*VW) {
					init = reduce_op(init, transform_op(concurrency::index<Rank>(y,x)));
				};
			};
#endif

			//---
			// Reduce through LDS across wavefront:
			int lx = localW * idx.local[0] + idx.local[1];
			tiled_data[lx] = init;
			BARRIER(waveSize);

			REDUCE_STEP(lx, 32);
			REDUCE_STEP(lx, 16);
			REDUCE_STEP(lx, 8);
			REDUCE_STEP(lx, 4);
			REDUCE_STEP(lx, 2);
			REDUCE_STEP(lx, 1);

			//---
			// Save result of this tile to global mem
			if (lx== 0) {
				results1[idx.tile[0]*computeUnits + idx.tile[1]] = tiled_data[0];
			};

		} );  //end parallel_for_each
		// results1[] now contains intermediate results which need to be combined together.

		//---
		//Copy partial array back to host
		// FIXME - we'd really like to use ZC memory for this final step
		//std::vector<outputT> h_data(resultCnt);
		//h_data = results1;
		concurrency::copy(*entry._dBuffer, *entry._stagingBuffer);



		outputT finalReduction = init;
		for (int i=0; i<results1.extent[0]; i++) {
			finalReduction = reduce_op(finalReduction, (*entry._stagingBuffer)[i]);
		};

#ifdef BOLT_POOL_ALLOC
		arrayPool.free(entry);
#endif

		return finalReduction;

	};

	template<typename outputT, int Rank, typename UnaryFunction, typename BinaryFunction>
	outputT transform_reduce_range(
		concurrency::index<Rank> origin, concurrency::extent<Rank> ext,
		UnaryFunction transform_op,
		outputT init,  BinaryFunction reduce_op)
	{
		return transform_reduce_range(concurrency::accelerator().get_default_view(), origin, ext, transform_op, init, reduce_op);
	};

	}; // end namespace amp

}; // end namespace bolt


#endif
