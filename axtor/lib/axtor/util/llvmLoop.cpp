/*
 * llvmLoop.cpp
 *
 *  Created on: Aug 6, 2010
 *      Author: simoll
 */

#include <axtor/util/llvmLoop.h>

namespace axtor {

	llvm::Loop * getNestedLoop(llvm::LoopInfo & loopInfo, llvm::Loop * parent, llvm::BasicBlock * block)
	{
		if (parent) { //get a directly nested subloop that contains @block if any
			const LoopVector & subLoops = parent->getSubLoops();

			for(LoopVector::const_iterator itLoop = subLoops.begin(); itLoop != subLoops.end(); ++itLoop)
			{
				llvm::Loop * subLoop = *itLoop;
				if (subLoop->contains(block))
				{
					return subLoop;
				}
			}

			return 0;

		} else { //return the outermost loop for this block
			llvm::Loop * innerMostLoop = loopInfo.getLoopFor(block);
			llvm::Loop * loop = innerMostLoop;

			while(loop != 0 && loop->getParentLoop() != 0)
			{
				loop = loop->getParentLoop();
			}

			return loop;
		}
	}

}
