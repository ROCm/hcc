/*  Axtor - AST-Extractor for LLVM
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
/*
 * NodeSplittingRestruct.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef NODESPLITTINGRESTRUCT_HPP_
#define NODESPLITTINGRESTRUCT_HPP_

#include "RestructuringProcedure.h"

#include <axtor/config.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmDuplication.h>

namespace axtor {

	/*
	 * acyclic restructuring procedure that uses a node splitting based algorithm
	 */
	class NodeSplittingRestruct : public RestructuringProcedure
	{

		static NodeSplittingRestruct instance;

		int findUniqueRegion(llvm::DominatorTree & domTree, RegionVector & regions, llvm::BasicBlock * block);

		BlockVector getAbstractSuccessors(llvm::BasicBlock * block, llvm::Loop * parentLoop, llvm::LoopInfo & loopInfo);

		/*
		 * split & rinse the top element
		 */
		llvm::BasicBlock * splitAndRinseOnStack(BlockVector & stack, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock * mandatoryExit);

		/*
		 * tries to generate a minimal number of splits for this block (one clone for each region)
		 */
		llvm::BasicBlock * splitAndRinseOnStack_perRegion(RegionVector & regions, BlockVector & stack, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock * mandatoryExit);

		/*
		 * sorts a given vector using the mutual node reachability as a relation
		 */
		void sortByReachability(BlockVector & vector, BlockSet regularExits);
		void sortByReachability(BlockVector & vector, int start, int end, BlockSet regularExits);

		/*
		 * returns a basic block that is unreachable from any other in @blocks
		 *
		 * this will always return a block if the reachability graph is acyclic
		 */
		llvm::BasicBlock * getUnreachableBlock(BlockSet blocks, BlockSet anticipated);

	public:
		NodeSplittingRestruct();
		~NodeSplittingRestruct();

		virtual bool resolve(RegionVector & regions, llvm::BasicBlock * requiredExit, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock* & oExitBlock);

		static NodeSplittingRestruct * getInstance();
	};
}


#endif /* NODESPLITTINGRESTRUCT_HPP_ */
